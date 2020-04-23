l_int clBuildProgram(cl_program program,
		     cl_uint num_devices,
		     const cl_device_id *device_list,
		     const char *options,
		     void (*pfn_notify)(cl_program, 
				     void *user_data),
		     void *user_data);
