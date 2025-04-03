kernel void histo(global const uchar* A, global int* H)
//function used from Tutorial 3 of Parallel Programming
{
	int id = get_global_id(0);

	int binIndex = A[id];
	//assumes that H has been initialised to location zero 
	atomic_inc(&H[binIndex]);
	//this atomic operation computes the histogram and stores the values at their buckets
	//atomic operations are very inefficient
}