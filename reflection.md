## Reflection Questions

0. Analyze your results, when does it make sense to use the various approaches?
- It makes sense to use CPU for smaller sizes, and GPU naive or utilizing cuBLAS when using larger input matrix sizes. 
1. How did your speed compare with cuBLAS?
- The GPU was faster than cuBLAS. Everything else was quite slower, especially at the larger sizes. 
2. What went well with this assignment?
- I liked having a lot of code explained in the slides. 
3. What was difficult?
- It was confusing to code the switching of arrays, but of course I could have just named the variables more informatively. 
4. How would you approach differently?
- Adding the timer code in initially. It gets boring and repetative. 
5. Anything else you want me to know?
- I liked seeing the different results based on different kernels with the same main. 
