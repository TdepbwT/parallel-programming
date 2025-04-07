#include <iostream>
#include <vector>

#include "include/Utils.h"
#include "include/CImg.h"

using namespace cimg_library;

void print_help() {
    std::cerr << "Application usage:" << std::endl;
    std::cerr << "  -p : select platform " << std::endl;
    std::cerr << "  -d : select device" << std::endl;
    std::cerr << "  -l : list all platforms and devices" << std::endl;
    std::cerr << "  -f : input image file (default: test.pgm)" << std::endl;
    std::cerr << "  -h : print this message" << std::endl;
}

int main(int argc, char** argv) {
    // Part 1 - handle command line options such as device selection, verbosity, etc.
    int platform_id = 0;
    int device_id = 0;
    std::string image_filename = "test.pgm";

    for (int i = 1; i < argc; i++) {
        if ((strcmp(argv[i], "-p") == 0) && (i < (argc - 1))) { platform_id = atoi(argv[++i]); }
        else if ((strcmp(argv[i], "-d") == 0) && (i < (argc - 1))) { device_id = atoi(argv[++i]); }
        else if (strcmp(argv[i], "-l") == 0) { std::cout << ListPlatformsDevices() << std::endl; }
        else if ((strcmp(argv[i], "-f") == 0) && (i < (argc - 1))) { image_filename = argv[++i]; }
        else if (strcmp(argv[i], "-h") == 0) { print_help(); return 0; }
    }

    cimg::exception_mode(0);



    // Detect any potential exceptions
    try {
        // Load input image
        CImg<unsigned char> image_input(image_filename.c_str());

        // Image validation
        if (image_input.is_empty()) {
            std::cerr << "Error: Failed to load image or image is empty." << std::endl;
            return 1;
        }

        std::cout << "Image loaded successfully: "
            << image_input.width() << "x" << image_input.height()
            << " with " << image_input.spectrum() << " channel(s)" << std::endl;

        CImgDisplay disp_input(image_input, ("Original: " + image_filename).c_str());
        int binSize = 256;  // Initialize bin size to 256 for 8-bit image

        // Host operations
        // use optimal work group size, let OpenCL choose
        size_t image_size = image_input.size();
        cl::NDRange global_size(image_size);
        cl::NDRange local_size = cl::NullRange;

        // Select the platform and device
        cl::Context context = GetContext(platform_id, device_id);

        // Display the selected device
        std::cout << "Running on " << GetPlatformName(platform_id) << ", " << GetDeviceName(platform_id, device_id) << std::endl;

        // Get device info to determine the optimal work group size
        cl::Device device = context.getInfo<CL_CONTEXT_DEVICES>()[0];
        size_t max_wg_size = device.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>();
        if (image_size > max_wg_size) {
            local_size = cl::NDRange(std::min(max_wg_size, (size_t)(256)));
            std::cout << "Setting local size to: " << std::min(max_wg_size, (size_t)(256)) << std::endl;
        }

        // Create a queue to which we will push commands for the device
        cl::CommandQueue queue(context, CL_QUEUE_PROFILING_ENABLE);

        // 3.2 Load & build the device code
        cl::Program::Sources sources;

        // Ensure the kernel source file path is correct
        AddSources(sources, "kernels/assessment_kernels.cl");
        cl::Program program(context, sources);

        // Build and debug the kernel code, throw errors if any
        try {
            program.build();
        }
        catch (const cl::Error& err) {
            std::cout << "Build Status: " << program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
            std::cout << "Build Options: " << program.getBuildInfo<CL_PROGRAM_BUILD_OPTIONS>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
            std::cout << "Build Log: " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
            throw err;
        }

        // Part 4 - allocate memory to histogram and cumulative histogram
        typedef int vec_type;

        // create histogram vector
        std::vector<vec_type> histogram(binSize, 0);
        size_t histogram_size = histogram.size() * sizeof(vec_type);

        // create cumulative histogram vector
        std::vector<vec_type> cum_histogram(binSize, 0);
        size_t cum_histogram_size = cum_histogram.size() * sizeof(vec_type);

        // create lookup table vector
        std::vector<vec_type> lookup(binSize, 0);
        size_t lookup_size = lookup.size() * sizeof(vec_type);

        std::cout << "Buffer sizes: Image=" << image_size << ", Histogram=" << histogram_size
            << ", CumHistogram=" << cum_histogram_size << ", Lookup=" << lookup_size << std::endl;

        // 4.1 Create buffers
        cl::Buffer buffer_image_input(context, CL_MEM_READ_ONLY, image_size);
        cl::Buffer buffer_histo_output(context, CL_MEM_READ_WRITE, histogram_size);
        cl::Buffer buffer_cum_histo_output(context, CL_MEM_READ_WRITE, cum_histogram_size);
        cl::Buffer buffer_lookup_output(context, CL_MEM_READ_WRITE, lookup_size);
        cl::Buffer buffer_image_output(context, CL_MEM_READ_WRITE, image_size);

        // 4.2 Copy image to device memory
        queue.enqueueWriteBuffer(buffer_image_input, CL_TRUE, 0, image_size, image_input.data());

		// add barrier to ensure write is complete
        queue.finish();

        // 4.3 Setup and execute the kernels for each step
        try {
            // ------- HISTOGRAM KERNEL -------
            cl::Kernel histogramKernel = cl::Kernel(program, "histogram");
			queue.enqueueFillBuffer(buffer_histo_output, 0, 0, histogram_size); // initialize clear histogram buffer

            //set args
            histogramKernel.setArg(0, buffer_image_input);
            histogramKernel.setArg(1, buffer_histo_output);
            histogramKernel.setArg(2, static_cast<int>(image_size));

            cl::Event histogram_event;
            queue.enqueueNDRangeKernel(histogramKernel, cl::NullRange, global_size, local_size, NULL, &histogram_event);
			histogram_event.wait(); // wait for kernel to finish
            queue.enqueueReadBuffer(buffer_histo_output, CL_TRUE, 0, histogram_size, histogram.data());

            std::cout << "Histogram kernel completed successfully" << std::endl;

            // ------- CUMULATIVE HISTOGRAM KERNEL -------
            queue.enqueueFillBuffer(buffer_cum_histo_output, 0, 0, cum_histogram_size);
            cl::Kernel cum_histogramKernel = cl::Kernel(program, "cumulative_histo");

            cum_histogramKernel.setArg(0, buffer_histo_output);
            cum_histogramKernel.setArg(1, buffer_cum_histo_output);
            cum_histogramKernel.setArg(2, binSize);

            cl::Event cum_histogram_event;
            queue.enqueueNDRangeKernel(cum_histogramKernel, cl::NullRange, cl::NDRange(1), cl::NullRange, NULL, &cum_histogram_event);
            cum_histogram_event.wait();
            queue.enqueueReadBuffer(buffer_cum_histo_output, CL_TRUE, 0, cum_histogram_size, cum_histogram.data());

            std::cout << "Cumulative histogram kernel completed successfully" << std::endl;

            // ------- LOOKUP TABLE KERNEL -------
            queue.enqueueFillBuffer(buffer_lookup_output, 0, 0, lookup_size);
            cl::Kernel lookupKernel = cl::Kernel(program, "lookuptable");

            lookupKernel.setArg(0, buffer_cum_histo_output);
            lookupKernel.setArg(1, buffer_lookup_output);
            lookupKernel.setArg(2, binSize);

            cl::Event lookup_event;
            queue.enqueueNDRangeKernel(lookupKernel, cl::NullRange, cl::NDRange(binSize), cl::NullRange, NULL, &lookup_event);
            lookup_event.wait();
            queue.enqueueReadBuffer(buffer_lookup_output, CL_TRUE, 0, lookup_size, lookup.data());

            std::cout << "Lookup table kernel completed successfully" << std::endl;

            // ------- IMAGE OUTPUT KERNEL -------
            cl::Kernel createimgKernel = cl::Kernel(program, "createimg");

            createimgKernel.setArg(0, buffer_image_input);
            createimgKernel.setArg(1, buffer_lookup_output);
            createimgKernel.setArg(2, buffer_image_output);
            createimgKernel.setArg(3, static_cast<int>(image_size));

            cl::Event createimg_event;
            queue.enqueueNDRangeKernel(createimgKernel, cl::NullRange, global_size, cl::NullRange, NULL, &createimg_event);
            createimg_event.wait();

            std::vector<unsigned char> buffer_image_output_vector(image_size);
            queue.enqueueReadBuffer(buffer_image_output, CL_TRUE, 0, image_size, buffer_image_output_vector.data());

            std::cout << "Create image kernel completed successfully" << std::endl;

            // Make sure all operations are finished
            queue.finish();

            // Display final normalized image
            CImg<unsigned char> output_image(buffer_image_output_vector.data(),
                image_input.width(),
                image_input.height(),
                image_input.depth(),
                image_input.spectrum());

            CImgDisplay disp_output(output_image, "Histogram Equalized Output");


            // Calculate processing time
			std::cout << "Processing time for histogram kernel: "
				<< (histogram_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - histogram_event.getProfilingInfo<CL_PROFILING_COMMAND_START>())
				<< " ns" << std::endl;

			std::cout << "Histogram memory transfer: " << GetFullProfilingInfo(histogram_event, PROF_US) << std::endl;

			std::cout << "Processing time for cumulative histogram kernel: "
				<< (cum_histogram_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - cum_histogram_event.getProfilingInfo<CL_PROFILING_COMMAND_START>())
				<< " ns" << std::endl;

            std::cout << "Cumulative histogram memory transfer: " << GetFullProfilingInfo(cum_histogram_event, PROF_US) << std::endl;

			std::cout << "Processing time for lookup table kernel: "
				<< (lookup_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - lookup_event.getProfilingInfo<CL_PROFILING_COMMAND_START>())
				<< " ns" << std::endl;

            std::cout << "Lookup table memory transfer: " << GetFullProfilingInfo(lookup_event, PROF_US) << std::endl;


			std::cout << "Processing time for image output kernel: "
				<< (createimg_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - createimg_event.getProfilingInfo<CL_PROFILING_COMMAND_START>())
				<< " ns" << std::endl;

            std::cout << "Image output memory transfer: " << GetFullProfilingInfo(createimg_event, PROF_US) << std::endl;


			double total_time =
				(histogram_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - histogram_event.getProfilingInfo<CL_PROFILING_COMMAND_START>())
				+ (cum_histogram_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - cum_histogram_event.getProfilingInfo<CL_PROFILING_COMMAND_START>())
				+ (lookup_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - lookup_event.getProfilingInfo<CL_PROFILING_COMMAND_START>())
				+ (createimg_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - createimg_event.getProfilingInfo<CL_PROFILING_COMMAND_START>());
			std::cout << "Total processing time: " << total_time << " ns" << std::endl;

            // Display images until closed
            unsigned int timeout_counter = 0;
            const unsigned int max_timeout = 300000; // 5 minutes at 1ms wait intervals

            while (!disp_input.is_closed() && !disp_output.is_closed()
                && !disp_input.is_keyESC() && !disp_output.is_keyESC()
                && timeout_counter < max_timeout) {
                disp_input.wait(1);
                disp_output.wait(1);
                timeout_counter++;
            }

        }
        catch (const cl::Error& err) {
            std::cerr << "OpenCL ERROR during kernel execution: " << err.what()
                << " (" << err.err() << ": " << getErrorString(err.err()) << ")" << std::endl;
            return 1;
        }

    }
    catch (const cl::Error& err) {
        std::cerr << "OpenCL ERROR: " << err.what() << " (" << err.err() << ": " << getErrorString(err.err()) << ")" << std::endl;
        return 1;
    }
    catch (CImgException& err) {
        std::cerr << "CImg ERROR: " << err.what() << std::endl;
        return 1;
    }
    catch (const std::exception& err) {
        std::cerr << "STANDARD ERROR: " << err.what() << std::endl;
        return 1;
    }
    catch (...) {
        std::cerr << "UNKNOWN ERROR occurred" << std::endl;
        return 1;
    }

    return 0;
}
