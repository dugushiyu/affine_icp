#define _CRT_SECURE_NO_WARNINGS
//#include<stdio.h>
#include <opencv/cv.h>
#include <opencv2/opencv.hpp>

#include <pcl/point_cloud.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/common/transforms.h>	
#include <iostream>
#include <vector>
using namespace std;
using namespace cv;
void remean(cv::Mat input, cv::Mat & output)
{
 cv::Mat mean;
 cv::reduce(input, mean, 0, CV_REDUCE_AVG);//4*3  1*3
 cv::Mat temp = cv::Mat::ones(input.rows, 1, CV_64FC1);//4*1
 output = input - temp*mean;
}


void remean(cv::Mat& input, cv::Mat& output, cv::Mat & mean)
{
 cv::reduce(input, mean, 0, CV_REDUCE_AVG);
 cv::Mat temp = cv::Mat::ones(input.rows, 1, CV_64FC1);
 output = input - temp*mean;
}


void whiten(cv::Mat input, cv::Mat &output)
{
 // need to be remean before whiten	
 const int N = input.rows;  //num of data
 const int M = input.cols;  //dimention	

 cv::Mat cov;
 cv::Mat D;
 cv::Mat E;
 cv::Mat temp = cv::Mat::eye(M, M, CV_64FC1);
 cv::Mat temp2;

 cov = input.t()*input / N;
 cv::eigen(cov, D, E);
 cv::sqrt(D, D);

 for (int i = 0; i < M; i++)
 {
  temp.at<double>(i, i) = D.at<double>(i, 0);
 }

 temp2 = E*temp.inv()*E.t()*input.t();
 output = temp2.t();
}

void whiten(cv::Mat input, cv::Mat &output, cv::Mat &E, cv::Mat &D)
{
 // need to be remean before whiten	
 const int N = input.rows;  //num of data
 const int M = input.cols;  //dimention	

 cv::Mat cov;
 cv::Mat D2;
 cv::Mat temp = cv::Mat::eye(M, M, CV_64FC1);
 cv::Mat temp2;
 cv::Mat E2;

 cov = input.t()*input / N;

 cv::eigen(cov, D, E2);
// std::cout << E2 << std::endl;
 cv::sqrt(D, D2);
 E = E2.t();

 for (int i = 0; i < M; i++)
 {
  temp.at<double>(i, i) = D2.at<double>(i, 0);
 }

 temp2 = E2*temp.inv()*E2.t()*input.t();
 output = temp2.t();
}

void runICA(cv::Mat input, cv::Mat &output, cv::Mat &W, int snum)     //output =Independent components matrix,W=Un-mixing matrix
{
 const  int M = input.rows;    // number of data
 const  int N = input.cols;    // data dimension

 const int maxIterations = 1000;
 const double epsilon = 0.0001;

 if (N < snum)
 {
  snum = M;
  printf(" Can't estimate more independent components than dimension of data ");
 }

 cv::Mat R(snum, N, CV_64FC1);
 cv::randn(R, cv::Scalar(0), cv::Scalar(1));
 cv::Mat ONE = cv::Mat::ones(M, 1, CV_64FC1);

 for (int i = 0; i < snum; ++i)
 {
  int iteration = 0;
  cv::Mat P(1, N, CV_64FC1);
  R.row(i).copyTo(P.row(0));

  while (iteration <= maxIterations)
  {
   iteration++;
   cv::Mat P2;
   P.copyTo(P2);
   cv::Mat temp1, temp2, temp3, temp4;
   temp1 = P*input.t();
   cv::pow(temp1, 3, temp2);
   cv::pow(temp1, 2, temp3);
   temp3 = 3 * temp3;
   temp4 = temp3*ONE;
   P = temp2*input / M - temp4*P / M;

   if (i != 0)
   {
    cv::Mat temp5;
    cv::Mat wj(1, N, CV_64FC1);
    cv::Mat temp6 = cv::Mat::zeros(1, N, CV_64FC1);

    for (int j = 0; j < i; ++j)
    {
     R.row(j).copyTo(wj.row(0));
     temp5 = P*wj.t()*wj;
     temp6 = temp6 + temp5;

    }
    P = P - temp6;
   }
   double Pnorm = cv::norm(P, 4);
   P = P / Pnorm;

   double j1 = cv::norm(P - P2, 4);
   double j2 = cv::norm(P + P2, 4);
   if (j1 < epsilon || j2 < epsilon)
   {
    P.row(0).copyTo(R.row(i));
    break;
   }
   else if (iteration == maxIterations)
   {
    P.row(0).copyTo(R.row(i));
   }
  }
 }
 output = R*input.t();
 W = R;
}

bool findNearTree(const pcl::PointCloud<pcl::PointXYZ> incloud, pcl::PointCloud<pcl::PointXYZ> searchcloud, pcl::PointCloud<pcl::PointXYZ>& outcloud){
 //searchcloud找到和incloud 一一对应的点
 pcl::KdTreeFLANN<pcl::PointXYZ> kdtree; //FLANN 快速近似近邻算法库实现的KdTree
 kdtree.setInputCloud(searchcloud.makeShared());

 int K = 1;
 std::vector<int> pointIdxNKNSearch(K);//保存邻近点索引 
 std::vector<float> pointNKNSquaredDistance(K);//保存对象点与邻近点的距离平方值
 for (int j = 0; j < incloud.points.size(); j++) {
  pcl::PointXYZ inpt;
  inpt.x = incloud[j].x;
  inpt.y = incloud[j].y;
  inpt.z = incloud[j].z;
  if (kdtree.nearestKSearch(inpt, K, pointIdxNKNSearch, pointNKNSquaredDistance) > 0) {
   for (std::size_t i = 0; i < pointIdxNKNSearch.size(); ++i) {
    pcl::PointXYZ tempt;
    tempt.x = searchcloud[pointIdxNKNSearch[i]].x;
    tempt.y = searchcloud[pointIdxNKNSearch[i]].y;
    tempt.z = searchcloud[pointIdxNKNSearch[i]].z;
    outcloud.points.push_back(tempt);
   }
  }
 }
 if (outcloud.points.size()<1) {
  return false;
 }
 return true;
}

bool besttransform(const Eigen::MatrixXf &A, const Eigen::MatrixXf &B, Eigen::Matrix4f& transformRT) {
 // affine registration of point sets using icp and ica 2008 shaoyidu naning zheng 公式是对的

 Eigen::MatrixXf centerA = A.rowwise().mean();
 Eigen::MatrixXf centerB = B.rowwise().mean();
 Eigen::MatrixXf subcenterM = A.colwise() - centerA.col(0);//m 3*4
 Eigen::MatrixXf subcenterN = B.colwise() - centerB.col(0);//n 3*4

 Eigen::MatrixXf AA = subcenterN*subcenterM.transpose();//3*3
 Eigen::MatrixXf NN = AA.rowwise().sum();//3*1

 Eigen::MatrixXf BB = subcenterM*subcenterM.transpose();//3*3
 Eigen::MatrixXf MM = BB.rowwise().sum();//3*1
 Eigen::MatrixXf ResR= AA*BB.inverse();// 3 * 3

 Eigen::MatrixXf AU = ResR*A;//3*4
 Eigen::MatrixXf AT = AU.rowwise().mean();
 Eigen::MatrixXf subMT = subcenterM.rowwise().mean();
 Eigen::MatrixXf ResT = centerB - AT;// subMT;//3*1

 transformRT.topLeftCorner(2, 2) = ResR;
 transformRT.topRightCorner(2, 1) = ResT;

 return 1;

}
int main2()
{
 cv::Mat m1 = (cv::Mat_<double>(4, 3) << 1, 5, 3, 4, 0, 6, 1, 8, 9, 100, 11, 12);
 cv::Mat d, e, w, s;
 remean(m1, m1);
 whiten(m1, m1, e, d);
 cout << m1 << endl;
 runICA(m1, s, w, m1.cols);

 //cout << w << endl;
// cout << s << endl;

 //
 Eigen::MatrixXf M(4, 3);
 M << 1, 5, 3, 4, 0, 6, 1, 8, 9, 100, 11, 12;
 Eigen::MatrixXf Mmean = M.colwise().mean();
 Eigen::MatrixXf MM = M.rowwise() - Mmean.row(0);
 Eigen::MatrixXf eigenM = MM.transpose()*MM / MM.rows();

 Eigen::EigenSolver<Eigen::MatrixXf> es(eigenM);
 Eigen::MatrixXcf eigenvalues = es.eigenvalues() ;
 Eigen::MatrixXcf eigenvectors = es.eigenvectors();
 Eigen::MatrixXf eigenval = eigenvalues.real();
 Eigen::MatrixXf eigenvalV = eigenval.array().sqrt();
 Eigen::Matrix3f tempEigen= Eigen::Matrix3f::Identity();;
 tempEigen(0, 0) = eigenvalV(0, 0);
 tempEigen(1, 1) = eigenvalV(1, 0);
 tempEigen(2, 2) = eigenvalV(2, 0);
 Eigen::Matrix3f eigenvectorsV;
 eigenvectorsV = eigenvectors.real();
 Eigen::MatrixXf output = (eigenvectorsV.transpose()*tempEigen.inverse()*eigenvectorsV*MM.transpose()).transpose();



 return 0;
}

bool readgif(string path, std::vector<cv::Mat>& frames)
{
 VideoCapture capture;
 bool flagfram = capture.open(path); //读取gif文件

 if (!capture.isOpened())
 {
  printf("can not open ...\n");
  return -1;
 }
 //存放gif的所有帧，每个frame都是Mat格式
 Mat frame;
 while (capture.read(frame))
 {
  frames.push_back(frame);
 }
 capture.release();
 return 1;
}
bool coutourmattocloud(cv::Mat inimg, pcl::PointCloud<pcl::PointXYZ>& incloud)
{
 cv::Mat threimg;
 cvtColor(inimg, threimg, CV_BGR2GRAY);
 threshold(threimg, threimg, 120, 255, THRESH_BINARY);
 vector<vector<Point> > contours;
 vector<Vec4i> hierarchy;
 findContours(threimg, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_NONE);

 ofstream outFile;
 outFile.open("E:\\0706\\07110_8.txt");

 for (int i = 0; i < contours.size(); i++)
 {
  for (int j = 0; j < contours[i].size(); j++)
  {
   pcl::PointXYZ tempt(0, 0, 0);
   tempt.x = contours[i][j].x;
   tempt.y = contours[i][j].y;

   incloud.points.push_back(tempt);
   outFile << tempt.x << "," << tempt.y << "," << tempt.z << endl;
  }
 }
 outFile.close();
 return 1;
}
int mainOK() {
 std::vector<cv::Mat> inimgv ;
 std::vector<cv::Mat> modelimgv;
 bool readflag=readgif("D:\\mpeg7shapeB\\original\\elephant-19.gif", inimgv);
  readflag = readgif("D:\\mpeg7shapeB\\original\\elephant-17.gif", modelimgv);

  cv::Mat inimg = inimgv[0];
  cv::Mat modelimg = modelimgv[0];

  pcl::PointCloud<pcl::PointXYZ> incloud;
  pcl::PointCloud<pcl::PointXYZ> modelcloud;
  coutourmattocloud(inimg, incloud);
  coutourmattocloud(modelimg, modelcloud);

 Eigen::Matrix4f tfRT = Eigen::Matrix4f::Identity();
 float rmse = 1000.0;
 float rmsepre = 0;
 int maxite = 100;
 int itenum = 0;

 while (itenum<maxite) {
  ofstream outFile;
  outFile.open("E:\\0706\\071110_9.txt");

 pcl::PointCloud<pcl::PointXYZ> inoutcloud;
 pcl::PointCloud<pcl::PointXYZ> modeloutcloud;

 // 查找最近点
 bool nearflag = findNearTree(incloud, modelcloud, modeloutcloud);
 nearflag = findNearTree(modelcloud,incloud, inoutcloud);
 // 合并最近点
 Eigen::MatrixXf inmatT = (incloud+inoutcloud).getMatrixXfMap();//n*4
 Eigen::MatrixXf modelmatT = (modeloutcloud + modelcloud).getMatrixXfMap();//n*4
 Eigen::MatrixXf inmat = inmatT.topLeftCorner(inmatT.rows()-2, inmatT.cols());
 Eigen::MatrixXf modelmat = modelmatT.topLeftCorner(modelmatT.rows() - 2, modelmatT.cols());

 //矩阵变换
 Eigen::Matrix4f transformRT = Eigen::Matrix4f::Identity();
 bool tb = besttransform(inmat, modelmat, transformRT);
 std::cout << transformRT << std::endl;
 tfRT = tfRT*transformRT;
 pcl::transformPointCloud(incloud, incloud, transformRT);
 for (size_t i = 0; i < incloud.points.size(); i++)
 {
  outFile << incloud.points[i].x << "," << incloud.points[i].y << "," << incloud.points[i].z << endl;
 }
  itenum++;
  cout <<"itenum="<<itenum<< "  rmse="<<rmse << endl;
  outFile.close();
 //求残差 squaredNorm()： inoutcloud---modelcloud  incloud--modeloutcloud  
  Eigen::MatrixXf intrcloudT = incloud.getMatrixXfMap() - modeloutcloud.getMatrixXfMap();
  Eigen::MatrixXf modeltrcloudT = inoutcloud.getMatrixXfMap() - modelcloud.getMatrixXfMap();
  Eigen::MatrixXf intrcloud = intrcloudT.topLeftCorner(intrcloudT.rows() - 2, intrcloudT.cols());
  Eigen::MatrixXf modeltrcloud = modeltrcloudT.topLeftCorner(modeltrcloudT.rows() - 2, modeltrcloudT.cols());
  float rmseTem = (intrcloud.squaredNorm() + modeltrcloud.squaredNorm())/(intrcloud.cols()+ modeltrcloud.cols());
  rmse = rmseTem-rmsepre;
  rmsepre = rmseTem;
  if (fabs(rmse)<0.001)
  {
   break;
  }

 }


 return 1;
}



bool readTxtFile(const string &fileName, const char tag,  pcl::PointCloud<pcl::PointXYZ> &pointCloud)
{
 cout << "reading file start..... " << endl;
 ifstream fin(fileName);
 string linestr;
 vector<pcl::PointXYZ> myPoint;
 while (getline(fin, linestr))
 {
  vector<string> strvec;
  string s;
  stringstream ss(linestr);
  while (getline(ss, s, tag))
  {
   strvec.push_back(s);
  }
  if (strvec.size() < 3) {
   cout << "格式不支持" << endl;
   return false;
  }
  pcl::PointXYZ p;
  p.x = stod(strvec[0]);
  p.y = stod(strvec[1]);
  p.z = stod(strvec[2]);
  myPoint.push_back(p);
  pointCloud.points.push_back(p);
 }
 fin.close();

 ////转换成pcd
 //pointCloud.width = (int)myPoint.size();
 //pointCloud.height = 1;
 //pointCloud.is_dense = false;
 //pointCloud.points.reserve(pointCloud.width * pointCloud.height);
 //for (int i = 0; i < myPoint.size(); i++)
 //{
 // pointCloud.points.push_back(myPoint[i]);
 //}
 cout << "reading file finished! " << endl;
 cout << "There are " << pointCloud.points.size() << " points!" << endl;
 return true;
}

int main() {
 //std::vector<cv::Mat> inimgv;
 //std::vector<cv::Mat> modelimgv;
 //bool readflag = readgif("D:\\mpeg7shapeB\\original\\elephant-19.gif", inimgv);
 //readflag = readgif("D:\\mpeg7shapeB\\original\\elephant-17.gif", modelimgv);

 //cv::Mat inimg = inimgv[0];
 //cv::Mat modelimg = modelimgv[0];

 pcl::PointCloud<pcl::PointXYZ> incloud;
 pcl::PointCloud<pcl::PointXYZ> modelcloud;
 //coutourmattocloud(inimg, incloud);
 //coutourmattocloud(modelimg, modelcloud);
 readTxtFile("E:\\zmData\\102602.txt", ',', incloud);
 readTxtFile("E:\\zmData\\102605.txt", ',', modelcloud);



 Eigen::Matrix4f tfRT = Eigen::Matrix4f::Identity();
 float rmse = 1000.0;
 float rmsepre = 0;
 int maxite = 100;
 int itenum = 0;

 while (itenum<maxite) {
  ofstream outFile;
  outFile.open("E:\\0706\\071110_9.txt");

  pcl::PointCloud<pcl::PointXYZ> inoutcloud;
  pcl::PointCloud<pcl::PointXYZ> modeloutcloud;

  // 查找最近点
  bool nearflag = findNearTree(incloud, modelcloud, modeloutcloud);
  nearflag = findNearTree(modelcloud, incloud, inoutcloud);
  // 合并最近点
  Eigen::MatrixXf inmatT = (incloud + inoutcloud).getMatrixXfMap();//n*4
  Eigen::MatrixXf modelmatT = (modeloutcloud + modelcloud).getMatrixXfMap();//n*4
  Eigen::MatrixXf inmat = inmatT.topLeftCorner(inmatT.rows() - 2, inmatT.cols());
  Eigen::MatrixXf modelmat = modelmatT.topLeftCorner(modelmatT.rows() - 2, modelmatT.cols());

  //矩阵变换
  Eigen::Matrix4f transformRT = Eigen::Matrix4f::Identity();
  bool tb = besttransform(inmat, modelmat, transformRT);
  std::cout << transformRT << std::endl;
  tfRT = tfRT*transformRT;
  pcl::transformPointCloud(incloud, incloud, transformRT);
  for (size_t i = 0; i < incloud.points.size(); i++)
  {
   outFile << incloud.points[i].x << "," << incloud.points[i].y << "," << incloud.points[i].z << endl;
  }
  itenum++;
  cout << "itenum=" << itenum << "  rmse=" << rmse << endl;
  outFile.close();
  //求残差 squaredNorm()： inoutcloud---modelcloud  incloud--modeloutcloud  
  Eigen::MatrixXf intrcloudT = incloud.getMatrixXfMap() - modeloutcloud.getMatrixXfMap();
  Eigen::MatrixXf modeltrcloudT = inoutcloud.getMatrixXfMap() - modelcloud.getMatrixXfMap();
  Eigen::MatrixXf intrcloud = intrcloudT.topLeftCorner(intrcloudT.rows() - 2, intrcloudT.cols());
  Eigen::MatrixXf modeltrcloud = modeltrcloudT.topLeftCorner(modeltrcloudT.rows() - 2, modeltrcloudT.cols());
  float rmseTem = (intrcloud.squaredNorm() + modeltrcloud.squaredNorm()) / (intrcloud.cols() + modeltrcloud.cols());
  rmse = rmseTem - rmsepre;
  rmsepre = rmseTem;
  if (fabs(rmse)<0.001)
  {
   break;
  }
 }

 return 1;
}