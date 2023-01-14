#pragma once

#include <utility>

#include <Eigen/Core>
#include <Eigen/Eigenvalues> 

namespace obb {
	struct ObbBox {
		Eigen::Vector3d position;
		Eigen::Vector3d axisX, axisY, axisZ;
		Eigen::Vector3d half_sizes;
	};

	inline ObbBox BuildObbBox(const Eigen::MatrixXi& face, const Eigen::MatrixXd& vertex) {
		assert(vertex.size() > 0);
		assert(face.size() > 0);
		auto n = face.rows();
		ObbBox box;
		Eigen::Vector3d position(0, 0, 0);
		for (auto i = 0; i < n; ++i) {
			position += vertex.row(face(i, 0)).transpose() + vertex.row(face(i, 1)).transpose() + vertex.row(face(i, 2)).transpose();
		}
		position /= 3 * n;
		box.position << position;

		// covariance matrix
		Eigen::Matrix3d C;
		for (auto i = 0; i < n; ++i) {
			Eigen::Vector3d p = vertex.row(face(i, 0)).transpose() - box.position;
			Eigen::Vector3d q = vertex.row(face(i, 1)).transpose() - box.position;
			Eigen::Vector3d r = vertex.row(face(i, 2)).transpose() - box.position;
			for (auto j = 0; j < 3; ++j) {
				for (auto k = 0; k < 3; ++k) {
					C(j, k) += p(j) * p(k) + q(j) * q(k) + r(j) * r(k);
				}
			}
		}
		C /= 3 * n;
		
		Eigen::EigenSolver<Eigen::Matrix3d> solver(C);
		auto eigenvectors = solver.eigenvectors();
		// auto eigenvalues = solver.eigenvalues();
		box.axisX << eigenvectors.col(0).real().normalized();
		box.axisY << eigenvectors.col(1).real().normalized();
		box.axisZ << eigenvectors.col(2).real().normalized();

		double max_x{ Eigen::Vector3d(vertex.row(0).transpose() - box.position).dot(box.axisX) },
			min_x{ max_x },
			max_y{ Eigen::Vector3d(vertex.row(0).transpose() - box.position).dot(box.axisY) },
			min_y{ max_y },
			max_z{ Eigen::Vector3d(vertex.row(0).transpose() - box.position).dot(box.axisZ) },
			min_z{ max_z };
		for (auto i = 1; i < vertex.rows(); ++i) {
			auto x = Eigen::Vector3d(vertex.row(i).transpose() - box.position).dot(box.axisX);
			auto y = Eigen::Vector3d(vertex.row(i).transpose() - box.position).dot(box.axisY);
			auto z = Eigen::Vector3d(vertex.row(i).transpose() - box.position).dot(box.axisZ);
			max_x = std::max(max_x, x);
			min_x = std::min(min_x, x);
			max_y = std::max(max_y, y);
			min_y = std::min(min_y, y);
			max_z = std::max(max_z, z);
			min_z = std::min(min_z, z);
		}

		box.half_sizes << (max_x - min_x) / 2.0, (max_y - min_y) / 2.0, (max_z - min_z) / 2.0;

		return box;
	}

} // namespace obb
