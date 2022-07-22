# NumPar: A `numpy` Clone with Parallelism Baked in

This is a term project for the course ICCS311: Functional and Parallel Programming by student Nawat Ngerncham.

Explanation video will be put [here](https://youtu.be/dQw4w9WgXcQ) once it has been done.

# Set up

In order to use the project, run the following commands:

```sh
# only has to set up once
python3 -m venv venv
source ./venv/bin/activate
pip install numpy maturin

# ensure that venv is activated
./run.sh
```

# A Quick Look at the Project

The full proposal can be found in `Proposal.pdf`

## Idea

This project aims to implement the following methods from `numpy` (as `np`) in Rust such that they will default to
running in parallel when possible or beneficial:

### Basic Vector Operations

- `np.dot`: Dot product of two arrays/vectors (Done)
- `np.linalg.norm`: Norm of a vector (Done)
- `np.outer`: Outer product of two arrays/vectors (Done, slow)
- ~~`np.tensordot`: Tensor dot product~~ (Too complicated)

### Basic Matrix Operations

- `np.linalg.trace`: Trace of matrix (Done)
- `np.transpose`: Matrix transpose (Done, slow)
- `np.linalg.det`: Determinant of matrix (Done, slow)
- `np.linalg.inv`: Inverse of matrix (Done, slow)

### Matrix Multiplication-related Operations

- `np.matmul` ~~or `@`~~: Matrix multiplication of matrices or matrix with arrays/vectors (Done, slow)
- `np.linalg.matrix_power`: Raising a matrix to a power (Done, slow)

### Matrix and System of Linear Equations-related Operations

- `np.linalg.solve`: Solve a matrix equation (Done, slow)
- `np.linalg.matrix_rank`: Rank of matrix (Done, slow)

Note that some of these may not be implemented by submission date due to ~~laziness~~ not having enough time to work.

## Measuring Success

We are measuring success using the running time of each function that will implemented compared to vanilla NumPy on
large data (suppose a function is for matrix computation, then the matrix would need to be quite large to minimize
parallelism overhead. If it can beat NumPy, great. If it cannot, then that's unfortunate.

The file `test.py` has some test cases for each of the implemented functions. Simply run the `run.sh` script to run the
test cases. Also, make sure that you have the right dependencies.

# Meme of the Repo

Me to `numpy`:

![](https://i.imgflip.com/420wbf.png)

# A more accurate Meme of the Repo

`numpy` to me when it comes to anything matrix-related (because I suck at memory optimization):

![](https://i.imgflip.com/420wbf.png)
