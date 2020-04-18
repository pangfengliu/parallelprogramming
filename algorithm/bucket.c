void bucketsort(int array[], int n, int b) 
{
  int i, j = 0;
  int *bucket = calloc(b + 1, sizeof(int));
  for (i = 0; i < n; i++)
    bucket[array[i]]++;
  for (i = 0; i <= b; i++)
    while(bucket[i]--)
      array[j++] = i;
}
