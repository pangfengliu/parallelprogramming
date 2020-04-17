#pragma omp parallel sections
{
#pragma section
  code 1;
#pragma section
  code 2;
}
/* the following will not compile */
#pragma omp parallel 
#pragma omp sections
#pragma omp section
  code 1;
#pragma omp section
  code 2;

