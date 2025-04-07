kernel void histogram(global const int* A, global int* H) { // histogram kernel taken from tutorial 3
	int id = get_global_id(0);

	//assumes that H has been initialised to 0
	int bin_index = A[id];//take value as a bin index

	atomic_inc(&H[bin_index]);//serial operation, not very efficient!
}

kernel void cumulative_histo(global const int*A, global int* cH, const int binSize)
{
	// calculates the cumulative histogram by using pointers to vectors, loops through until it has reached the binsize
	int id = get_global_id(0);

	for (int i = id + 1; i < binSize && id < binSize; i++)
	{
		atomic_add(&cH[i], A[id]/3);
	}
}

kernel void lookuptable(global const int* A, global int* B) // guess what this does
{
	//create lookup value with values for each pixel
	int id = get_global_id(0);

	B[id] = A[id] * (double)255 / A[255];
}

// kernel to adjust input image with normalised histogram from lookup table
// casts onto image to produce output image

kernel void createimg(global uchar* A, global int* lookup, global uchar* nImg)
{
	int id = get_global_id(0);
	//create new image with normalised histogram
	nImg[id] = (uchar)lookup[A[id]];
}