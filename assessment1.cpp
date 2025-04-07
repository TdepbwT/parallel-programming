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
        CImg<unsigned char> image_input(image_filename.c_str());
        CImg<unsigned char> input_image_8; // Create an 8-bit image
        CImgDisplay disp_input(image_input, "test.pgm");
        int binSize = 256;  // Initialize bin size to 256, which is an 8-bit image

        // Host operations

        // Select the platform and device
        cl::Context context = GetContext(platform_id, device_id);

        // Display the selected device
        std::cout << "Running on " << GetPlatformName(platform_id) << ", " << GetDeviceName(platform_id, device_id) << std::endl;

        // Create a queue to which we will push commands for the device
        cl::CommandQueue queue(context);

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
            std::cout << "Build Options:\t" << program.getBuildInfo<CL_PROGRAM_BUILD_OPTIONS>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
            std::cout << "Build Log:\t " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
            throw err;
        }

        // Part 4 - allocate memory to histogram and cumulative histogram

        // Create type for vector
        typedef int vec_type;
        // Create histogram vector
        std::vector<vec_type> histogram(binSize, 0); // Histogram vector
        // Get size
        size_t histogram_size = histogram.size() * sizeof(vec_type);

        // Create cumulative histogram vector
        std::vector<vec_type> cum_histogram(binSize, 0); // Cumulative histogram vector
        // Get size
        size_t cum_histogram_size = cum_histogram.size() * sizeof(vec_type);

        // Create lookup table vector
        std::vector<vec_type> lookup(binSize, 0); // Lookup table vector
        // Get size
        size_t lookup_size = lookup.size() * sizeof(vec_type);

        // 4.1 Create buffers for image input and output
        cl::Buffer buffer_image_input(context, CL_MEM_READ_ONLY, image_input.size()); // Image buffer
        cl::Buffer buffer_histo_output(context, CL_MEM_READ_WRITE, histogram_size); // Histogram buffer
        cl::Buffer buffer_cum_histo_output(context, CL_MEM_READ_WRITE, cum_histogram_size); // Cumulative histogram buffer
        cl::Buffer buffer_lookup_output(context, CL_MEM_READ_WRITE, lookup_size); // Lookup table buffer
        cl::Buffer buffer_image_output(context, CL_MEM_READ_WRITE, image_input.size()); // Output image buffer

        // 4.2 Copy images to device memory
        queue.enqueueWriteBuffer(buffer_image_input, CL_TRUE, 0, image_input.size(), &image_input.data()[0]);

        // 4.3 Setup and execute the kernels for each step

        // Histogram kernel
        cl::Kernel histogramKernel = cl::Kernel(program, "histogram");
        histogramKernel.setArg(0, buffer_image_input);
        histogramKernel.setArg(1, buffer_histo_output);
        cl::Event histogram_event;
        queue.enqueueNDRangeKernel(histogramKernel, cl::NullRange, cl::NDRange(image_input.size()), cl::NullRange, NULL, &histogram_event);
        queue.enqueueReadBuffer(buffer_histo_output, CL_TRUE, 0, histogram_size, &histogram);

        // Cumulative histogram kernel
        queue.enqueueFillBuffer(buffer_cum_histo_output, 0, 0, cum_histogram_size);
        cl::Kernel cum_histogramKernel = cl::Kernel(program, "cumulative_histo");
        cum_histogramKernel.setArg(0, buffer_histo_output);
        cum_histogramKernel.setArg(1, buffer_cum_histo_output);
        cum_histogramKernel.setArg(2, binSize);
        cl::Event cum_histogram_event;
        queue.enqueueNDRangeKernel(cum_histogramKernel, cl::NullRange, cl::NDRange(binSize), cl::NullRange, NULL, &cum_histogram_event);
        queue.enqueueReadBuffer(buffer_cum_histo_output, CL_TRUE, 0, cum_histogram_size, &cum_histogram[0]);

        // Lookup table kernel
        queue.enqueueFillBuffer(buffer_lookup_output, 0, 0, lookup_size);
        cl::Kernel lookupKernel = cl::Kernel(program, "lookuptable");
        lookupKernel.setArg(0, buffer_cum_histo_output);
        lookupKernel.setArg(1, buffer_lookup_output);
        cl::Event lookup_event;
        queue.enqueueNDRangeKernel(lookupKernel, cl::NullRange, cl::NDRange(lookup_size), cl::NullRange, NULL, &lookup_event);
        queue.enqueueReadBuffer(buffer_lookup_output, CL_TRUE, 0, lookup_size, &lookup[0]);

        // Image output kernel
        cl::Kernel createimgKernel = cl::Kernel(program, "createimg");
        createimgKernel.setArg(0, buffer_image_input);
        createimgKernel.setArg(1, buffer_lookup_output);
        createimgKernel.setArg(2, buffer_image_output);
        cl::Event createimg_event;
        queue.enqueueNDRangeKernel(createimgKernel, cl::NullRange, cl::NDRange(image_input.size()), cl::NullRange, NULL, &createimg_event);
        std::vector<unsigned char> buffer_image_output_vector(image_input.size());
        queue.enqueueReadBuffer(buffer_image_output, CL_TRUE, 0, buffer_image_output_vector.size(), buffer_image_output_vector.data());

        // Display final normalized image
        CImg<unsigned char> output_image(buffer_image_output_vector.data(), image_input.width(), image_input.height(), image_input.depth(), image_input.spectrum());
        CImgDisplay disp_output(output_image, "output");

        while (!disp_input.is_closed() && !disp_output.is_closed()
            && !disp_input.is_keyESC() && !disp_output.is_keyESC()) {
            disp_input.wait(1);
            disp_output.wait(1);
        }
    }
    catch (const cl::Error& err) {
        std::cerr << "ERROR: " << err.what() << ", " << getErrorString(err.err()) << std::endl;
    }
    catch (CImgException& err) {
        std::cerr << "ERROR: " << err.what() << std::endl;
    }

    return 0;
}
