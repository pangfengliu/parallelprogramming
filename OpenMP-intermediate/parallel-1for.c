#pragma omp parallel 
#pragma omp for
  for (...)
    code;
/* The following is equivlent */
#pragma omp parallel for
  for (...)
    code;


