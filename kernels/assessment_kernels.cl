kernel void histogram(global const uchar* A, global int* H, const int size) { // histogram kernel, tutorial 3 used as base
	int id = get_global_id(0);

	if (id < size) {
		//assumes that H has been initialised to 0
		int bin_index = A[id];//take value as a bin index

		atomic_inc(&H[bin_index]);//serial operation, not very efficient!
	}
}

kernel void cumulative_histo(global const int* A, global int* cH, const int binSize) {
	int id = get_global_id(0);

	// copy first bin from histogram
	if (id == 0) {
		cH[0] = A[0];
		// calculate cumulative sum
		for (int i = 1; i < binSize; i++) {
			cH[i] = cH[i - 1] + A[i];
		}
	}
}

kernel void lookuptable(global const int* A, global int* B, const int binSize) {
	// create lookup value with values for each pixel
	int id = get_global_id(0);

	if (id < binSize) {
		// avoid division by zero
		if (A[binSize - 1] > 0) {
			B[id] = (int)((float)A[id] * 255.0f / A[binSize - 1]);
		}
		else {
			B[id] = 0;
		}
	}
}


// kernel to adjust input image with normalised histogram from lookup table
// casts onto image to produce output image

kernel void createimg(global uchar* A, global int* lookup, global uchar* nImg, const int size) {
	int id = get_global_id(0);

	if (id < size) {
		// create new image with normalised histogram
		nImg[id] = (uchar)lookup[A[id]];
	}
}