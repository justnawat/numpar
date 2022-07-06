# NumPar: A `numpy` Clone with Parallelism Baked in

This is a term project for the course ICCS311: Functional and Parallel Programming by student Nawat Ngerncham.

Explanation video will be put [here](https://youtu.be/dQw4w9WgXcQ) once it has been done.

# A Quick Look at the Project

The full proposal can be found in `Proposal.pdf`

## Idea

This project aims to implement the following methods from `numpy` (as `np`) in Rust such that they will default to running in parallel when possible or beneficial:

- `np.dot`: Dot product of two arrays/vectors
- `np.outer`: Outer product of two arrays/vectors
- `np.matmul` or `@`: Matrix multiplication of matrices or matrix with arrays/vectors
- `np.tensordot`: Tensor dot product
- `np.linalg.matrix_power`: Raising a matrix to a power
- `np.linalg.norm`: Norm of a vector
- `np.linalg.det`: Determinant of matrix
- `np.linalg.matrix_rank`: Rank of matrix
- `np.linalg.trace`: Trace of matrix
- `np.linalg.solve`: Solve a matrix equation
- `np.linalg.inv`: Inverse of matrix

Note that some of these may not be implemented by submission date due to ~~laziness~~ not having enough time to work.

## Measuring Success

We are measuring success using the running time of each function that will implemented compared to vanilla NumPy on large data (suppose a function is for matrix computation, then the matrix would need to be quite large to minimize parallelism overhead. If it can beat NumPy, great. If it cannot, then that's unfortunate.

# Meme of the Repo

Me to `numpy`:

![](https://i.imgflip.com/420wbf.png)
