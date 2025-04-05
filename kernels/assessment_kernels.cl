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

kernel void lookuptable(global const int* input, global const int* lookup, global int* output, const int size) // guess what this does
{
	int id = get_global_id(0);

	if (id < size)
	{
		int index = input[id];
		output[id] = lookup[index];
	}
}