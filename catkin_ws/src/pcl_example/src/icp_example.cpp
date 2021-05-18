#include <iostream>
#include <unistd.h>
#include <vector>
#include <ros/ros.h>
#include <boost/foreach.hpp>
#include <vector>
#include <Eigen/Core>
#include <string>
#include <sensor_msgs/PointCloud2.h>

// PCL library
#include <pcl_ros/point_cloud.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/features/normal_3d.h>
#include <pcl/io/ply_io.h>
#include <pcl/common/transforms.h>
#include <pcl/common/common.h>
#include <pcl/common/centroid.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/conditional_removal.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/registration/icp.h>
typedef pcl::PointCloud<pcl::PointXYZRGB> PointCloudXYZRGB;
typedef pcl::PointCloud<pcl::PointXYZRGBNormal> PointCloudXYZRGBNormal;
using namespace std;

PointCloudXYZRGB::Ptr sub_cloud(new PointCloudXYZRGB);

void addNormal(PointCloudXYZRGB::Ptr cloud, PointCloudXYZRGBNormal::Ptr cloud_with_normals)
{
  /*Add normal to PointXYZRGB
		Args:
			cloud: PointCloudXYZRGB
			cloud_with_normals: PointXYZRGBNormal
	*/
  pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
  pcl::search::KdTree<pcl::PointXYZRGB>::Ptr searchTree(new pcl::search::KdTree<pcl::PointXYZRGB>);
  searchTree->setInputCloud(cloud);
  pcl::NormalEstimation<pcl::PointXYZRGB, pcl::Normal> normalEstimator;
  normalEstimator.setInputCloud(cloud);
  normalEstimator.setSearchMethod(searchTree);
  //normalEstimator.setKSearch ( 50 );
  normalEstimator.setRadiusSearch(0.01);
  normalEstimator.compute(*normals);
  pcl::concatenateFields(*cloud, *normals, *cloud_with_normals);
  vector<int> indices;
  pcl::removeNaNNormalsFromPointCloud(*cloud_with_normals, *cloud_with_normals, indices);
  return;
}

void point_preprocess(PointCloudXYZRGB::Ptr cloud)
{
  /*Preprocess point before ICP
	Args:
		cloud: PointCloudXYZRGB
	*/
  //////////////Step1. Remove Nan////////////
  printf("Original point number: %d\n", cloud->points.size());
  vector<int> indices;
  pcl::removeNaNFromPointCloud(*cloud, *cloud, indices);

  //////////////Step2. Downsample////////////
  pcl::VoxelGrid<pcl::PointXYZRGB> sor;
  sor.setInputCloud(cloud);
  sor.setLeafSize(0.004f, 0.004f, 0.004f);
  sor.filter(*cloud);
  copyPointCloud(*cloud, *cloud);
  printf("Downsampled point number: %d\n", cloud->points.size());

  //////////////Step3. Denoise//////////////
  pcl::StatisticalOutlierRemoval<pcl::PointXYZRGB> sor2;
  if (cloud->points.size() > 100)
  {
    sor2.setInputCloud(cloud);
    sor2.setMeanK(50);
    sor2.setStddevMulThresh(0.5);
    sor2.filter(*cloud);
  }
  vector<int> indices2;
  pcl::removeNaNFromPointCloud(*cloud, *cloud, indices2);
  printf("Donoised point number: %d\n", cloud->points.size());

  // build the condition
  pcl::ConditionAnd<pcl::PointXYZRGB>::Ptr range_cond(new pcl::ConditionAnd<pcl::PointXYZRGB>());
  range_cond->addComparison(pcl::FieldComparison<pcl::PointXYZRGB>::ConstPtr(new pcl::FieldComparison<pcl::PointXYZRGB>("x", pcl::ComparisonOps::GT, -0.08)));
  range_cond->addComparison(pcl::FieldComparison<pcl::PointXYZRGB>::ConstPtr(new pcl::FieldComparison<pcl::PointXYZRGB>("x", pcl::ComparisonOps::LT, 0.13)));
  range_cond->addComparison(pcl::FieldComparison<pcl::PointXYZRGB>::ConstPtr(new pcl::FieldComparison<pcl::PointXYZRGB>("y", pcl::ComparisonOps::GT, -0.06)));
  range_cond->addComparison(pcl::FieldComparison<pcl::PointXYZRGB>::ConstPtr(new pcl::FieldComparison<pcl::PointXYZRGB>("y", pcl::ComparisonOps::LT, 0.11)));
  range_cond->addComparison(pcl::FieldComparison<pcl::PointXYZRGB>::ConstPtr(new pcl::FieldComparison<pcl::PointXYZRGB>("z", pcl::ComparisonOps::GT, -0.1)));
  range_cond->addComparison(pcl::FieldComparison<pcl::PointXYZRGB>::ConstPtr(new pcl::FieldComparison<pcl::PointXYZRGB>("z", pcl::ComparisonOps::LT, 0.65)));
  // build the filter
  pcl::ConditionalRemoval<pcl::PointXYZRGB> condrem;
  condrem.setCondition(range_cond);
  condrem.setInputCloud(cloud);
  condrem.setKeepOrganized(true);
  // apply filter
  condrem.filter(*cloud);

  return;
}

Eigen::Matrix4f initial_guess(PointCloudXYZRGB::Ptr cloud_src, PointCloudXYZRGB::Ptr cloud_target)
{
  Eigen::Vector4f src_centroid, target_centroid;
  pcl::compute3DCentroid(*cloud_src, src_centroid);
  pcl::compute3DCentroid(*cloud_target, target_centroid);
  Eigen::Matrix4f tf_tran = Eigen::Matrix4f::Identity();
  Eigen::Matrix4f tf_rot = Eigen::Matrix4f::Identity();

  Eigen::Matrix3f covariance;
  pcl::computeCovarianceMatrixNormalized(*cloud_src, src_centroid, covariance);
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> eigen_solver(covariance, Eigen::ComputeEigenvectors);
  Eigen::Matrix3f eigenVectorsPCA = eigen_solver.eigenvectors();
  Eigen::Vector3f eigenValuesPCA = eigen_solver.eigenvalues();

  Eigen::Matrix3f covariance2;
  pcl::computeCovarianceMatrixNormalized(*cloud_target, target_centroid, covariance2);
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> eigen_solver2(covariance2, Eigen::ComputeEigenvectors);
  Eigen::Matrix3f eigenVectorsPCA2 = eigen_solver2.eigenvectors();

  // Eigen::Quaternion<float> rot_q = Eigen::Quaternion<float>::FromTwoVectors(eigenVectorsPCA.row(0),eigenVectorsPCA2.row(0));
  Eigen::Matrix3f R;
  R = eigenVectorsPCA2 * eigenVectorsPCA.inverse();
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++)
      tf_rot(i, j) = R(i, j);

  tf_tran(0, 3) = target_centroid[0] - src_centroid[0];
  tf_tran(1, 3) = target_centroid[1] - src_centroid[1];
  tf_tran(2, 3) = target_centroid[2] - src_centroid[2];
  Eigen::Matrix4f tf = tf_rot * tf_tran;
  return tf;
}

Eigen::Matrix4f point_2_plane_icp(PointCloudXYZRGB::Ptr cloud_src, PointCloudXYZRGB::Ptr cloud_target, PointCloudXYZRGBNormal::Ptr trans_cloud)
{
  PointCloudXYZRGBNormal::Ptr cloud_source_normals(new PointCloudXYZRGBNormal);
  PointCloudXYZRGBNormal::Ptr cloud_target_normals(new PointCloudXYZRGBNormal);
  addNormal(cloud_src, cloud_source_normals);
  addNormal(cloud_target, cloud_target_normals);
  pcl::IterativeClosestPointWithNormals<pcl::PointXYZRGBNormal, pcl::PointXYZRGBNormal>::Ptr icp(new pcl::IterativeClosestPointWithNormals<pcl::PointXYZRGBNormal, pcl::PointXYZRGBNormal>());
  icp->setMaximumIterations(2000);
  icp->setMaxCorrespondenceDistance(0.7);
  icp->setTransformationEpsilon(1e-10);
  icp->setEuclideanFitnessEpsilon(0.001);
  icp->setInputSource(cloud_source_normals); // not cloud_source, but cloud_source_trans!
  icp->setInputTarget(cloud_target_normals);

  // registration
  icp->align(*trans_cloud); // use cloud with normals for ICP

  if (icp->hasConverged())
  {
    cout << "icp score: " << icp->getFitnessScore() << endl;
  }
  else
    cout << "Not converged." << endl;
  Eigen::Matrix4f inverse_transformation = icp->getFinalTransformation();
}

Eigen::Matrix4f point_2_point_icp(PointCloudXYZRGB::Ptr cloud_src, PointCloudXYZRGB::Ptr cloud_target, PointCloudXYZRGB::Ptr trans_cloud)
{
  pcl::IterativeClosestPoint<pcl::PointXYZRGB, pcl::PointXYZRGB>::Ptr icp(new pcl::IterativeClosestPoint<pcl::PointXYZRGB, pcl::PointXYZRGB>());
  icp->setMaximumIterations(2000);
  //icp->setMaxCorrespondenceDistance(0.7);
  icp->setTransformationEpsilon(1e-10);
  icp->setEuclideanFitnessEpsilon(0.001);
  icp->setInputSource(cloud_src); // not cloud_source, but cloud_source_trans!
  icp->setInputTarget(cloud_target);

  // registration
  icp->align(*trans_cloud); // use cloud with normals for ICP

  if (icp->hasConverged())
  {
    cout << "icp score: " << icp->getFitnessScore() << endl;
  }
  else
    cout << "Not converged." << endl;
  Eigen::Matrix4f inverse_transformation = icp->getFinalTransformation();
}

void cloud_cb(const sensor_msgs::PointCloud2ConstPtr &input)
{
  PointCloudXYZRGB::Ptr cloud(new PointCloudXYZRGB);
  pcl::fromROSMsg(*input, *cloud); //convert from PointCloud2 to pcl point type
  //Exmaple : pcl PointCloudXYZRGB information
  printf("-------------------------Cloud information-----------------------------\n");
  printf("Original Cloud size: %d\n", cloud->points.size());
  int cloud_size = cloud->points.size();
  printf("The first cloud coordinate and color information:\n");
  printf("X: %4lf, Y: %4lf, Z: %4lf, R: %d, G: %d, B: %d\n", cloud->points[0].x, cloud->points[0].y, cloud->points[0].z, cloud->points[0].r, cloud->points[0].g, cloud->points[0].b);
  printf("The last cloud coordinate and color information:\n");
  printf("X: %4lf, Y: %4lf, Z: %4lf, R: %d, G: %d, B: %d\n", cloud->points[cloud_size - 1].x, cloud->points[cloud_size - 1].y, cloud->points[cloud_size - 1].z, cloud->points[cloud_size - 1].r, cloud->points[cloud_size - 1].g, cloud->points[cloud_size - 1].b);

  std::vector<int> indices;
  PointCloudXYZRGB::Ptr filtered_cloud(new PointCloudXYZRGB);
  pcl::removeNaNFromPointCloud(*cloud, *filtered_cloud, indices);
  printf("Nonnan Cloud Number: %d\n", filtered_cloud->points.size());
  printf("**********************************************************************\n");
  *sub_cloud = *filtered_cloud;
  point_preprocess(sub_cloud);
  usleep(200);
}

int main(int argc, char **argv)
{
  ros::init(argc, argv, "icp");
  ros::NodeHandle nh;

  ros::Publisher model_publisher = nh.advertise<sensor_msgs::PointCloud2>("/camera/model", 1);
  ros::Publisher cloud_publisher = nh.advertise<sensor_msgs::PointCloud2>("/camera/cloud", 1);
  ros::Publisher initial_guess_tf_publisher = nh.advertise<sensor_msgs::PointCloud2>("/camera/ini_guess", 1);
  ros::Publisher registered_cloud_publisher = nh.advertise<sensor_msgs::PointCloud2>("/camera/registered_cloud", 1);
  ros::Subscriber model_subscriber = nh.subscribe<sensor_msgs::PointCloud2>("/camera/depth_registered/points", 10, cloud_cb);

  /*Load pre-scanned Model and observed cloud*/
  PointCloudXYZRGB::Ptr model(new PointCloudXYZRGB);
  //PointCloudXYZRGB::Ptr cloud(new PointCloudXYZRGB);
  PointCloudXYZRGB::Ptr ini_guess_tf_cloud(new PointCloudXYZRGB);
  PointCloudXYZRGBNormal::Ptr registered_cloud_normal(new PointCloudXYZRGBNormal);
  PointCloudXYZRGB::Ptr registered_cloud(new PointCloudXYZRGB);
  string model_path;
  string cloud_path;
  nh.getParam("model_path", model_path);
  nh.getParam("cloud_path", cloud_path);

  printf("Load model\n");
  pcl::io::loadPCDFile<pcl::PointXYZRGB>(model_path, *model);
  printf("Finish Load pointcloud of model\n");
  //pcl::PolygonMesh mesh;
  //pcl::io::loadPLYFile(cloud_path, mesh);
  //pcl::fromPCLPointCloud2(mesh.cloud, *cloud);
  //printf("Finish Load pointcloud of cloud\n");

  model->header.frame_id = "/map";
  sub_cloud->header.frame_id = "/map";

  //point_preprocess(model);
  /*
  printf("Initial guess\n");
  Eigen::Matrix4f tf1 = initial_guess(sub_cloud, model);

  pcl::transformPointCloud(*sub_cloud, *ini_guess_tf_cloud, tf1);
  printf("ICP\n");
  Eigen::Matrix4f tf2 = point_2_plane_icp(ini_guess_tf_cloud, model, registered_cloud_normal);
  Eigen::Matrix4f final_tf = tf1 * tf2;
  cout << final_tf << endl;
  */
  while (ros::ok())
  {
    //printf("Load model\n");
    //pcl::io::loadPCDFile<pcl::PointXYZRGB>(model_path, *model);
    //printf("Finish Load pointcloud of model\n");

    model->header.frame_id = "/map";
    sub_cloud->header.frame_id = "/map";

    //point_preprocess(model);

    printf("Initial guess\n");
    Eigen::Matrix4f tf1 = initial_guess(model, sub_cloud);

    pcl::transformPointCloud(*model, *ini_guess_tf_cloud, tf1);
    printf("ICP\n");
    Eigen::Matrix4f tf2 = point_2_point_icp(ini_guess_tf_cloud, sub_cloud, registered_cloud);
    Eigen::Matrix4f final_tf = tf1 * tf2;
    cout << final_tf << endl;

    model_publisher.publish(model);
    cloud_publisher.publish(sub_cloud);
    initial_guess_tf_publisher.publish(ini_guess_tf_cloud);
    registered_cloud_publisher.publish(registered_cloud);

    ros::spinOnce();
  }

  return 0;
}
