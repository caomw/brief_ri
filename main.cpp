#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <string>
#include <algorithm>

#include <cmath>
#include <cstring>
#include "brief_ri.h"

using namespace std;
using namespace cv;


static const int FAST_THRESH = 20;
static const int MAX_FAST_FEATURES = 500;
static const string dataDir("/home/feixh/OpenCV/opencv-2.4.10/samples/cpp/fabmap/");
static const string imgroot("/home/feixh/workspace/Data/image/disjoint/");
int main(int argc, char **argv){
	
	if ( argc != 5 ){
		cerr << "need 3 args" << endl;
		return -1;
	}
	ostringstream ss;
	ss << argv[1];
	string im1_file( imgroot + ss.str() + ".jpg" );
	cout << im1_file << endl;
	ss.str("");
	ss.clear();

	ss << argv[2];
	string im2_file( imgroot + ss.str() + ".jpg" );
	cout << im2_file << endl;
	ss.str("");
	ss.clear();


	Mat im1, im2;
	im1 = imread( im1_file, CV_LOAD_IMAGE_GRAYSCALE );
	im2 = imread( im2_file, CV_LOAD_IMAGE_GRAYSCALE );
	if ( im1.empty() || im2.empty() ){
		cerr << "cannot open image files" << endl;
		return -1;
	}

    BriefRIDescriptorExtractor extractor1;
    extractor1.rotatePattern( stof( argv[3] ) );

    BriefRIDescriptorExtractor extractor2;
    extractor2.rotatePattern( stof( argv[4] ) );

    Ptr<DescriptorMatcher> matcher = new BFMatcher( NORM_HAMMING, true );

	vector< KeyPoint > kp1;
	FAST( im1, kp1, FAST_THRESH );
	sort( kp1.begin(), kp1.end(), []( const KeyPoint &x, const KeyPoint &y){ return x.response > y.response; } );
	kp1.resize( MAX_FAST_FEATURES );
	Mat desc1;
	extractor1.compute( im1, kp1, desc1 );

	vector< KeyPoint > kp2;
	FAST( im2, kp2, FAST_THRESH );
	sort( kp2.begin(), kp2.end(), []( const KeyPoint &x, const KeyPoint &y){ return x.response > y.response; } );
	kp2.resize( MAX_FAST_FEATURES );
	Mat desc2;
	extractor2.compute( im2, kp2, desc2 );

	vector< DMatch > matches;
	matcher->match( desc1, desc2, matches );

	vector< Point2f > pts1, pts2;
	for ( const auto &it : matches ){
		pts1.push_back( kp1[ it.queryIdx ].pt );
		pts2.push_back( kp2[ it.trainIdx ].pt );
	}
	Mat status( 1, matches.size(), CV_8UC1 );
	findFundamentalMat( pts1, pts2, CV_FM_RANSAC, 3.0, 0.99, status );
	cout << status << endl;
	
	// vector< DMatch > matches_inl;
	// int count = 0;
	// for ( int i = 0; i < status.cols; ++i ){
	// 	if ( status.at< char >( i ) ){
	// 		matches_inl.push_back( matches[ i ] );
	// 		++count;
	// 	}
	// }
	// cout << "inlier ratio=" << count / (float) status.cols << endl;
	vector< DMatch > matches_inl( matches.begin(), matches.end() );

	Mat disp;
	drawMatches( im1, kp1, im2, kp2, matches_inl, disp );
	imshow( "matches", disp );
	waitKey();

	int t_count = 0;
	int f_count = 0;
	for ( int i = 0; i < matches_inl.size(); ++i ){
		vector< DMatch > tmp;
		tmp.push_back( matches_inl[ i ] );
		drawMatches( im1, kp1, im2, kp2, tmp, disp );
		imshow( "matches", disp );
		char c;
		c = waitKey();
		if ( c == 'y' ){
			++t_count;
		}else{
			if ( c == 'n' ){
				++f_count;
			}
		}
	}
	cout << "count/total=" << t_count / (float) ( t_count + f_count ) << endl;

}

