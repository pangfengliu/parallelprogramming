 cl_int 
 clEnqueueReadBuffer(cl_command_queue command_queue,
		     cl_mem buffer,
		     cl_bool blocking_read,
		     size_t offset,
		     size_t cb,
		     void *ptr,
		     cl_uint num_events_in_wait_list,
		     const cl_event *event_wait_list,
		     cl_event *event);
