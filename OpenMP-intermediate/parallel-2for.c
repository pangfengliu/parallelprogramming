#pragma omp parallel for
  for (...)
#pragma omp paralell for
  for (...)

#pragma omp parallel 
  {
#pragma omp for
    for (...)
#pragma omp for
    for (...)
  }

