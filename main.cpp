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
static const string imgroot("./");

static const int SUPPORT_SIZE = 61;
static const int HALF_SUPPORT_SIZE = ( SUPPORT_SIZE >> 1 );
bool getRotatedPatch( const Mat &img, Mat &cropped, const KeyPoint& pt, float angle = 0 ){
	if ( angle== 0 ){

		// no rotation
		if ( pt.pt.x + HALF_SUPPORT_SIZE >= img.cols || pt.pt.x - HALF_SUPPORT_SIZE < 0 || pt.pt.y + HALF_SUPPORT_SIZE >= img.rows || pt.pt.y - HALF_SUPPORT_SIZE < 0 ){
			return false;
		}else{
			getRectSubPix( img, Size( SUPPORT_SIZE, SUPPORT_SIZE ), pt.pt, cropped );
			return true;
		}
	}else{
		RotatedRect rRect = RotatedRect( pt.pt, Size2f( SUPPORT_SIZE, SUPPORT_SIZE ), angle );
		// get the bounding box of the rotated patch
		Rect bRect = rRect.boundingRect();
		Point2f center( bRect.width/2, bRect.height/2 );
		if ( bRect.x < 0 || bRect.x + bRect.width >= img.cols || bRect.y < 0 || bRect.y + bRect.height >= img.rows ){
			return false;
		}
		// get the content of the bounding box
		Mat roi( img, bRect );
		// get rotation matrix
		Mat M = getRotationMatrix2D( center, angle, 1.0 );
		Mat rotated; 
		warpAffine( roi, rotated, M, roi.size(), INTER_LINEAR );
		getRectSubPix( rotated, rRect.size, center, cropped );
// #define _DEBUG
#ifdef _DEBUG
		imshow("roi", roi );
		waitKey();
		imshow("rotated", rotated );
		waitKey();
		imshow("cropped", cropped );
		waitKey();
#endif
		return true;
	}

}

void computeBriefRI( const Mat &im2, vector< KeyPoint > &kp2, Mat &desc2, float angle ){
	// when the image is aligned to the gravity, angle = 0
	// rotate anti-clockwise, angle > 0
	// rotate clockwise, angle < 0
	BriefDescriptorExtractor extractor;
	vector< KeyPoint > kp2_t;
	for ( auto kp : kp2 ){
		Mat patch;
		Mat desc;
		vector< KeyPoint > tmpKp;
		if ( getRotatedPatch( im2, patch, kp, -angle ) ){  // notice, here is a negative angle
			tmpKp.push_back( KeyPoint( HALF_SUPPORT_SIZE, HALF_SUPPORT_SIZE, 1 ) );
			extractor.compute( patch, tmpKp, desc );
			if ( ! tmpKp.empty() ){
				kp2_t.push_back( kp );
				desc2.push_back( desc );
				cout << desc << endl;
			}else{
				cout << "deleted" << endl;
			}
		}
	}
	kp2 = kp2_t;
}
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

	float t1 = stof( argv[3] );
	float t2 = stof( argv[4] );


	Mat im1, im2;
	im1 = imread( im1_file, CV_LOAD_IMAGE_GRAYSCALE );
	im2 = imread( im2_file, CV_LOAD_IMAGE_GRAYSCALE );
	if ( im1.empty() || im2.empty() ){
		cerr << "cannot open image files" << endl;
		return -1;
	}

    // BriefRIDescriptorExtractor extractor1( t1 );
    // BriefRIDescriptorExtractor extractor2( t2 );
	BriefDescriptorExtractor extractor;

    Ptr<DescriptorMatcher> matcher = new BFMatcher( NORM_HAMMING, true );

	vector< KeyPoint > kp1;
	FAST( im1, kp1, FAST_THRESH );
	sort( kp1.begin(), kp1.end(), []( const KeyPoint &x, const KeyPoint &y){ return x.response > y.response; } );
	kp1.resize( MAX_FAST_FEATURES );
	Mat desc1;
	computeBriefRI( im1, kp1, desc1, t1 );

	vector< KeyPoint > kp2;
	FAST( im2, kp2, FAST_THRESH );
	sort( kp2.begin(), kp2.end(), []( const KeyPoint &x, const KeyPoint &y){ return x.response > y.response; } );
	kp2.resize( MAX_FAST_FEATURES );
	Mat desc2;
	computeBriefRI( im2, kp2, desc2, t2 );


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
	
	vector< DMatch > matches_inl;
	int count = 0;
	for ( int i = 0; i < status.cols; ++i ){
		if ( status.at< char >( i ) ){
			matches_inl.push_back( matches[ i ] );
			++count;
		}
	}
	cout << "inlier ratio=" << count / (float) status.cols << endl;
	// vector< DMatch > matches_inl( matches.begin(), matches.end() );

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
	cout << "total=" << t_count + f_count << endl;
	cout << "count/total=" << t_count / (float) ( t_count + f_count ) << endl;

}

