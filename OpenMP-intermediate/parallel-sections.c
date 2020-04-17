#pragma omp parallel sections
{
#pragma section
  code 1;
#pragma section
  code 2;
}
/* The following is equivlent */
#pragma omp parallel 
{
#pragma omp sections
  {
#pragma omp section
    code 1;
#pragma omp section
    code 2;
  }
}

