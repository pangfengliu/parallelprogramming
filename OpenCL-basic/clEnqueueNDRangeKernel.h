cl_int 
clEnqueueNDRangeKernel (cl_command_queue command_queue,
			cl_kernel kernel,
			cl_uint work_dim,
			const size_t *global_work_offset,
			const size_t *global_work_size,
			const size_t *local_work_size,
			cl_uint num_events_in_wait_list,
			const cl_event *event_wait_list,
			cl_event *event);
