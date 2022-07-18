#include <Eigen/Core>
#include <Eigen/Dense>
#include <iostream>
#include <ctime>
#include <random>
#include <cstdlib>

using namespace std;
using namespace Eigen;

// find the next binary vector
bool next(VectorXd &v) {
	for (int i = 0; i < v.size(); ++i) {
		if (v(i) == 0) {
			v(i) = 1;
			return true;
		} else {
			v(i) = 0;
		}
	}
	return false;
}

// check if a vector is binary
bool check_binary(VectorXd v) {
	VectorXd v0 = v.cwiseAbs();
	VectorXd v1 = (v - VectorXd::Ones(v.size())).cwiseAbs();
	if (v0.cwiseMin(v1).maxCoeff() < 1e-6) {
		return true;
	} else {
		return false;
	}
}

void attack_equ(MatrixXd A, int n, int d) {
	// binary vector
	VectorXd b = VectorXd::Zero(d);
	// d x d submatrix
	MatrixXd X = A(seqN(0, d), all);
	auto Lu = X.partialPivLu();
	while (next(b)) {
		VectorXd x = Lu.solve(b);
		VectorXd v = A * x;
		if (check_binary(v)) {
			cout << v << endl;
		}
	}
}

// convert a vector to binary
void convert_binary(VectorXd &v) {
	for (int i = 0; i < v.size(); ++i) {
		if (v(i) > 0.5) {
			v(i) = 1;
		} else {
			v(i) = 0;
		}
	}
}

void attack_reg(MatrixXd A, int n, int d) {
	// choose r rows
	int r = d + 1;
	// leverage score sampling
	BDCSVD<MatrixXd> svd(A, ComputeThinU | ComputeThinV);
	MatrixXd U = svd.matrixU();
	VectorXd P = U.array().square().rowwise().sum();
	vector<double> p(&P[0], P.data() + P.cols() * P.rows());
	
	discrete_distribution<> dist(p.begin(), p.end());
	default_random_engine rng;
	vector<int> rows;
	for (int i = 0; i < r; ++i) {
		rows.push_back(dist(rng));
	}
	
	MatrixXd X = A(rows, all);
	for (int i = 0; i < r; ++i) {
		X(i, all) /= sqrt(r * p[rows[i]]);
	}
	// binary vector
	VectorXd b = VectorXd::Zero(r);
	// linear regression solver
	auto normal_X = (X.transpose() * X).ldlt();
	auto normal_A = (A.transpose() * A).ldlt();
	double opt_err = -1;
	VectorXd opt_vec;
	while (next(b)) {
		VectorXd x = normal_X.solve(X.transpose() * b);
		VectorXd v = A * x;
		convert_binary(v);
		x = normal_A.solve(A.transpose() * v);
		double err = (A * x - v).array().square().sum();
		if (err < opt_err || opt_err < 0) {
			opt_err = err;
			opt_vec = v;
		}
	}
	//cout << opt_err << endl;
	//cout << opt_vec << endl;
}

int main(int argc, char** argv) {
	// number of data records and dimension
	int n, d;
	
	n = atoi(argv[1]);
	d = atoi(argv[2]);
	
	// generate a random matrix of size n x d
	MatrixXd A;
	A = MatrixXd::Random(n, d);
	
	double start_time = clock();
	
	if (strcmp(argv[3], "equ") == 0) {
		// attack by solving linear equations
		attack_equ(A, n, d);
	}
	
	if (strcmp(argv[3], "reg") == 0) {	
		// attack by solving linear regression
		attack_reg(A, n, d);
	}
	
	double end_time = clock();
	cout << (end_time - start_time) / 1000 << " seconds" << endl;
	return 0;
}
