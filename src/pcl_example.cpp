#include <ros/ros.h>
#include <Eigen/Core>
// PCL specific includes
#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/common/transforms.h>
#include "pcl_ros/point_cloud.h"
#include <pcl/filters/filter.h>

typedef pcl::PointCloud<pcl::PointXYZRGB> PointCloudXYZRGB;

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

  Eigen::Affine3f transform_2 = Eigen::Affine3f::Identity();
  // Define a translation of 2.5 meters on the x axis.
  transform_2.translation() << -0.630, 0.04, 0.02;

  float theta = M_PI/2;

  // The same rotation matrix as before; theta radians around X,Y axis
  transform_2.rotate (Eigen::AngleAxisf (-theta, Eigen::Vector3f::UnitX()));
  transform_2.rotate (Eigen::AngleAxisf (theta, Eigen::Vector3f::UnitY()));

  // Print the transformation
  printf ("\nMethod #2: using an Affine3f\n");
  std::cout << transform_2.matrix() << std::endl;

  // Executing the transformation
  PointCloudXYZRGB::Ptr transformed_cloud (new PointCloudXYZRGB);
  // You can either apply transform_1 or transform_2; they are the same
  pcl::transformPointCloud (*filtered_cloud, *transformed_cloud, transform_2);

  pcl::io::savePCDFileASCII ("/home/jacky/pcl_icp/catkin_ws/test_pcd.pcd", *transformed_cloud);
  printf("Nonnan Cloud Number: %d\n", filtered_cloud->points.size());
  printf("**********************************************************************\n");
}

int main(int argc, char **argv)
{
  // Initialize ROS
  ros::init(argc, argv, "my_pcl_tutorial");
  ros::NodeHandle nh;
  // Create a ROS subscriber for the input point cloud
  ros::Subscriber model_subscriber = nh.subscribe<sensor_msgs::PointCloud2>("/cloud_pcd", 1, cloud_cb);

  // Spin
  ros::spin();
}
