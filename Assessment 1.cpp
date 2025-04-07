/*
 CMP3752M - Parallel Programming
David A - 26321509
This program uses code adapted from https://github.com/charles-fox/OpenCL-Tutorials. The base template was tutorial 2.cpp
The program uses kernels that use atomic operations that were used from tutorial 3 to execute on the input image; this can be changed using the -f flag.
This program will firstly transform an image into a histogram, which is then transformed into a cumulative histogram using atomic functions as well. This then gets put into a lookup table where it gets normalised and then uses the final kernel to transform the input image into a normalised image that will be outputted by the end. The execution times and memory transfers are recorded as well as total execution time.
*/

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
	//Part 1 - handle command line options such as device selection, verbosity, etc.
	int platform_id = 0;
	int device_id = 0;
	string image_filename = "test.pgm";

	for (int i = 1; i < argc; i++) {
		if ((strcmp(argv[i], "-p") == 0) && (i < (argc - 1))) { platform_id = atoi(argv[++i]); }
		else if ((strcmp(argv[i], "-d") == 0) && (i < (argc - 1))) { device_id = atoi(argv[++i]); }
		else if (strcmp(argv[i], "-l") == 0) { std::cout << ListPlatformsDevices() << std::endl; }
		else if ((strcmp(argv[i], "-f") == 0) && (i < (argc - 1))) { image_filename = argv[++i]; }
		else if (strcmp(argv[i], "-h") == 0) { print_help(); return 0; }
	}

	cimg::exception_mode(0);

	//detect any potential exceptions
	try {
		CImg<unsigned char> image_input(image_filename.c_str());
		CImg<unsigned char> input_image_8; //create an 8 bit image
		CImgDisplay disp_input(image_input, "test.pgm");
		int binSize = 256;  //initialise binsize to 256, which is an 8 bit image

		//host operations

		//select the platform and device
		cl::Context context = GetContext(platform_id, device_id);

		//display the selected device
		std::cout << "Running on " << GetPlatformName(platform_id) << ", " << GetDeviceName(platform_id, device_id) << std::endl;

		//create a queue to which we will push commands for the device
		cl::CommandQueue queue(context);

		//3.2 Load & build the device code
		cl::Program::Sources sources;

		AddSources(sources, "kernels/my_kernels.cl");

		cl::Program program(context, sources);

		//build and debug the kernel code, throw errors if any
		try {
			program.build();
		}
		catch (const cl::Error& err) {
			std::cout << "Build Status: " << program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			std::cout << "Build Options:\t" << program.getBuildInfo<CL_PROGRAM_BUILD_OPTIONS>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			std::cout << "Build Log:\t " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			throw err;
		}

		//Part 4 - allocate memory to histogram and cumulative histogram

		// create type for vector
		typedef int vec_type;
		//create histogram vector
		std::vector<vec_type> histogram(binSize, 0); // histogram vector
		//get size
		size_t histogram_size = histogram.size() * sizeof(vec_type);

		// create cumulative histogram vector
		std::vector<vec_type> cum_histogram(binSize, 0); // cumulative histogram vector
		//get size
		size_t cum_histogram_size = cum_histogram.size() * sizeof(vec_type);

		// create lookup table vector
		std::vector<vec_type> lookup(binSize, 0); // lookup table vector
		//get size
		size_t lookup_size = lookup.size() * sizeof(vec_type);

		
		//4.1 create buffers for image input and output
		cl::Buffer buffer_image_input(context, CL_MEM_READ_ONLY, image_input.size()); // image buffer

		cl::Buffer buffer_histo_output(context, CL_MEM_READ_WRITE, histogram_size); // histogram buffer

		cl::Buffer buffer_cum_histo_output(context, CL_MEM_READ_WRITE, cum_histogram_size); // cumulative histogram buffer

		cl::Buffer buffer_lookup_output(context, CL_MEM_READ_WRITE, lookup_size); // lookup table buffer

		cl::Buffer buffer_image_output(context, CL_MEM_READ_WRITE, image_input.size()); // output image buffer


		//4.2 Copy images to device memory
		queue.enqueueWriteBuffer(buffer_image_input, CL_TRUE, 0, image_input.size(), &image_input.data()[0]);
	

		// 4.3 Setup and execute the kernels for each step

		//histogram kernel

		// create histogram kernel to work out the histogram of the image
		cl::Kernel histogramKernel = cl::Kernel(program, "histogram");

		// set kernel arguments to take in image and output histogram vector
		histogramKernel.setArg(0, buffer_image_input);
		histogramKernel.setArg(1, buffer_histo_output);

		// create event for histogram kernel
		cl::Event histogram_event;

		// add kernel to queue
		queue.enqueueNDRangeKernel(histogramKernel, cl::NullRange, cl::NDRange(image_input.size()), cl::NullRange, NULL, &histogram_event);
		queue.enqueueReadBuffer(buffer_histo_output, CL_TRUE, 0, histogram_size, &histogram);

		// read result into output buffer
		queue.enqueueFillBuffer(buffer_cum_histo_output, 0, 0, cum_histogram_size);

		//set up cumulative histogram kernel
		cl::Kernel cum_histogramKernel = cl::Kernel(program, "cumulative_histo");

		// set kernel arguments to take in histogram and output cumulative histogram vector
		cum_histogramKernel.setArg(0, buffer_histo_output);
		cum_histogramKernel.setArg(1, buffer_cum_histo_output);
		cum_histogramKernel.setArg(2, binSize);

		// create event for cumulative histogram kernel
		cl::Event cum_histogram_event;
		// add kernel to queue
		queue.enqueueNDRangeKernel(cum_histogramKernel, cl::NullRange, cl::NDRange(binSize), cl::NullRange, NULL, &cum_histogram_event);
		queue.enqueueReadBuffer(buffer_cum_histo_output, CL_TRUE, 0, cum_histogram_size, &cum_histogram[0]);

		// read result into output buffer
		queue.enqueueFillBuffer(buffer_lookup_output, 0, 0, lookup_size);

		// event for lookup kernel
		cl::Kernel lookupKernel = cl::Kernel(program, "lookuptable");

		// set kernel arguments to take in cumulative histogram and output lookup table vector
		lookupKernel.setArg(0, buffer_cum_histo_output);
		lookupKernel.setArg(1, buffer_lookup_output);

		// event for lookup kernel, add to queue
		cl::Event lookup_event;
		queue.enqueueNDRangeKernel(lookupKernel, cl::NullRange, cl::NDRange(lookup_size), cl::NullRange, NULL, &lookup_event);
		queue.enqueueReadBuffer(buffer_lookup_output, CL_TRUE, 0, lookup_size, &lookup[0]);

		// event for image output
		cl::Kernel createimgKernel = cl::Kernel(program, "createimg");

		// setup normalised histogram through lookup table
		createimgKernel.setArg(0, buffer_image_input);
		createimgKernel.setArg(1, buffer_lookup_output);
		createimgKernel.setArg(2, buffer_image_output);

		// setup image output

		cl::Event createimg_event;
		queue.enqueueNDRangeKernel(createimgKernel, cl::NullRange, cl::NDRange(image_input.size()), cl::NullRange, NULL, &createimg_event);
		vector<unsigned char> buffer_image_output_vector(image_input.size());
		queue.enqueueReadBuffer(buffer_image_output, CL_TRUE, 0, buffer_image_output_vector.size(), &buffer_image_output_vector.data()[0]);






		// display final normalised image
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
