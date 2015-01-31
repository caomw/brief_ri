#include "brief_ri.h"
#include "cmath"

using namespace cv;
using namespace std;
#define PI 3.1415926
int pattern_[] = {
-1,-2,-1,7,-1,-14,3,-3,-2,1,2,11,6,1,-7,-10,2,13,0,-1,5,-14,-3,5,8,-2,4,2,8,-11,5,-15,-23,-6,-9,8,6,-12,8,-10,-1,-3,1,8,6,3,6,5,-6,-7,-5,5,-2,22,-8,-11,7,14,5,8,14,-1,-14,-5,9,-14,0,2,-3,7,6,22,6,-6,-5,-8,9,-5,-1,7,-7,-3,-18,-10,-5,4,11,0,3,2,10,9,3,-10,9,4,12,0,19,-3,15,1,-5,-11,-1,14,8,7,-23,7,5,-5,-6,0,17,-10,-4,13,-4,-3,1,-12,2,-12,8,0,22,3,13,-13,-1,3,17,-16,10,6,15,7,0,-5,-12,2,-2,19,-6,3,-15,-4,3,8,14,0,-11,4,5,5,-7,11,1,7,12,6,3,21,2,-3,1,14,1,5,11,-5,-17,3,2,-6,8,6,-10,5,-2,-14,4,0,-7,5,5,-6,4,10,-7,4,0,22,-18,7,-3,-1,18,0,22,-4,3,-5,-7,1,-3,2,-20,19,-2,17,-10,3,24,-8,-14,-5,5,7,12,-2,-15,-4,12,4,-19,0,13,20,5,3,-12,-8,0,5,6,-5,-11,-7,-11,6,-22,-3,4,15,1,10,-4,-7,-6,15,10,5,24,0,6,3,-2,22,14,-13,-4,4,8,-13,-22,-18,-1,-1,3,-7,-12,-19,3,4,10,8,-2,13,-1,-6,-5,-6,-21,2,2,-3,-7,4,16,0,-5,-6,-1,-12,-1,1,18,9,10,-7,6,-11,3,4,-7,19,5,-18,5,-4,0,4,4,-20,-11,7,12,18,17,-20,7,-18,15,2,-11,19,6,-18,3,-7,1,-4,13,-14,3,17,-8,2,2,-7,6,1,-9,17,8,-2,-6,-8,12,-1,4,-2,6,-1,7,-2,8,6,-1,-8,-9,-7,-9,8,0,15,22,0,-15,-4,-1,-14,-2,3,-4,-7,-7,17,-2,-8,-4,9,-7,5,7,7,13,-5,11,-8,-4,11,8,0,-11,5,-6,-9,-6,2,-20,3,2,-6,10,6,-6,-6,7,-15,-3,-6,1,2,0,11,2,-3,-12,7,5,14,-7,0,-1,-1,0,-16,8,6,11,22,-3,0,0,19,-17,5,-14,-23,-19,-13,10,-8,-2,-11,6,-11,13,-10,-7,1,0,14,1,-12,-5,-5,7,4,-1,8,-5,-1,2,15,-1,-3,-10,7,-6,3,-18,10,-13,-7,10,-13,-1,1,-10,13,14,-19,-14,8,-13,-4,1,7,-2,1,-7,12,-5,3,-5,1,-2,-2,-10,8,14,2,7,8,9,3,2,8,1,-9,0,-18,0,4,12,1,9,0,-10,-14,-9,-13,6,-2,5,1,10,10,-6,-3,-5,-16,6,11,0,-5,10,-23,2,1,-5,13,9,-3,-1,-4,-5,-13,13,10,8,-11,20,19,2,-9,-8,4,-9,0,10,-14,19,15,-12,-14,-3,-10,-3,-23,-2,17,-11,-3,-14,6,-2,19,2,-4,5,-5,-13,3,-2,2,4,-5,4,17,-11,17,-2,-7,23,1,13,8,-16,1,-5,-13,-17,1,6,4,-3,-8,-9,-5,-10,-2,0,-9,-2,-7,0,5,2,5,-16,-4,3,6,-15,2,12,-2,-1,4,2,6,1,1,-8,-2,12,-2,-2,-5,8,-8,9,-9,-10,2,1,3,10,-4,4,-9,12,6,5,2,-8,-3,5,0,1,-13,2,-7,-10,-1,-18,7,8,-1,-10,-9,-1,-23,2,6,-3,-5,2,3,11,0,-7,-4,2,15,-3,-10,-8,-20,3,-13,-12,-19,-11,5,-13,-17,2,-3,4,7,0,-12,-1,5,-6,-14,11,-4,-4,0,10,3,-3,7,21,13,6,-11,24,-12,-4,-7,16,4,-14,3,5,-3,-12,-7,-4,0,-5,7,-9,-17,-7,13,-6,22,5,-11,-8,2,-11,23,-10,7,14,-1,-10,-3,3,8,1,-13,0,-6,-21,-7,-14,6,19,18,-6,-4,7,10,-4,-1,21,-1,-5,1,6,-10,-2,-11,-3,18,7,-1,-9,-3,10,-5,14,-13,-3,17,-19,11,-18,-1,-2,8,-23,-18,-5,0,-9,-2,-11,-4,-8,2,6,14,-6,-3,0,-3,0,-15,4,-9,-9,-15,11,-1,11,3,-16,-10,7,-7,-10,-2,-2,-10,-3,-5,-23,5,-8,13,-11,-15,11,-15,-6,6,-3,-16,2,-2,12,6,24,-16,0,-10,11,8,7,-7,-7,-19,16,5,-3,9,7,9,-16,-7,2,3,9,-10,1,21,7,8,0,7,17,1,12,-8,6,9,-7,11,-6,-8,0,19,3,9,-7,1,-11,-5,8,0,14,-2,-2,12,-6,-15,12,4,-21,0,-4,17,-7,-6,-9,-10,-7,-14,-10,-15,-14,-15,-5,-7,-12,5,0,-4,-4,15,2,5,-23,-6,-21,-4,4,-6,5,-10,6,-15,-3,4,5,-1,19,-4,-4,-23,17,-4,-11,13,12,1,-14,4,-6,-11,10,-20,5,4,20,3,-20,-8,1,3,9,-19,-3,9,15,18,-4,11,16,12,7,8,-8,-14,9,-3,0,-6,-4,2,-10,1,2,-1,-7,8,18,-6,12,9,-23,-7,-6,8,2,5,6,-9,-7,-12,-2,-1,2,-7,9,9,15,7,2,6,6,-6,12,16,19,0,3,4,0,6,-1,-2,17,2,1,8,1,3,-1,-12,0,-11,2,-11,9,7,3,-1,4,-19,-11,-1,3,-1,-10,1,-4,-10,3,-2,11,6,7,3,-8,-9,-14,24,-10,-2,-3,-3,-6,-18,-10,-13,-1,-7,-7,2,-6,9,-4,2,-13,6,-4,4,3,-2,2,-4,13,9,5,-11,-11,-6,-2,4,-9,11,0,-19,-5,-23,-7,-5,-6,-3,-4,-6,14,12,-11,12,-16,-8,15,-21,6,-12,-1,-2,16,-8,-1,6,-2,-8,-1,1,8,-9,-4,3,-2,-2,0,-7,-8,4,-11,11,2,-12,3,2,7,11,-4,-7,-6,-9,-7,3,0,-5,-7,3,-5,-10,-1,-3,-10,8,8,0,1,5,0,9,16,1,4,8,-3,-11,9,-15,17,8,2,0,17,-9,-11,-6,-3,-10,1,1,-8,15,-13,-12,4,-2,4,-6,-10,-6,-7,5,-5,7,6,10,9,8,7,-5,-3,-18,3,-6,4,5,-13,-10,-3,-5,2,-11,0,-16,-21,7,-13,-5,-14,-14,-4,-4,9,4,-3,7,11,4,-4,10,17,6,17,9,8,-10,-11,0,-16,-6,8,-6,5,-13,-5,10,2,3,16,12,-8,13,-6,0,0,10,-11,4,5,8,-2,10,-7,11,3,-13,4,2,-3,-7,-2,-14,16,-11,-6,11,6,7,15,-3,-10,8,8,-3,-12,12,6,-13,7,-14,-5,-11,-6,-8,-6,7,3,6,10,-4,1,5,16,9,13,10,10,-17,8,2,1,-5,-4,4,8,-14,2,-5,-9,4,-3,-6,-7,3,0,-10,-8,-2,4,-10,5,-8,24,-9,-8,2,-9,8,17,-4,2,-5,0,14,9,-9,15,11,5,-6,1,-8,4,-3,-21,9,2,10,-1,2,11,4,3,24,-2,2,17,-8,-10,-14,5,6,7,-13,10,11,-1,0,6,4,6,-10,-2,-12,6,5,-1,3,-15,8,-4,1,11,-7,11,1,0,5,-12,6,1,10,-2,-3,4,-1,-11,-2,12,-1,-8,7,-18,-20,0,2,2,-9,-1,-13,2,-16,-1,3,-17,-5,8,15,-14,3,-12,-13,15,6,-8,2,6,2,22,6,-23,-3,-7,-2,0,-6,-10,13,6,-6,7,6,12,-10,7,-6,11,-2,-22,0,-17,-2,-1,-4,-14,-11,-8,-2,12,7,-5,12,-13,7,-2,2,6,-7,8,0,23,-3,12,6,-11,13,-10,-21,8,10,0,-3,15,7,-6,7,-12,-5,-10,-21,-11,12,-11,-5,-11,8,0,5,-1,-11,-9,8,-1,7,-23,11,-5,21,-5,0,6,-8,8,-6,12,8,5,-7,-2,3,-20,-5,9,-12,12,-6,3,-11,5,4,11,13,12,2,-12,13,-13,-4,7,4,15,0,-16,-3,2,-3,14,-2,-14,4,-11,16,3,-13,10,23,-19,9,5,2,3,5,-7,14,-13,19,15,-11,0,14,-5,-2,-4,11,-6,0,5,-2,-8,-13,-15,-11,-17,-7,3,1,-8,-10,-10,-13,-12,7,-13,0,-6,23,-17,2,-3,-7,3,1,-10,4,4,13,-6,14,-2,-19,5,-1,-8,9,-5,10,-1,7,7,5,-10,9,0,19,5,7,-7,-4,1,-11,-11,-1,-1,2,11,-4,7,-1,-2,2,-20,1,-6,-9,-18,-4,-18,8,-2,-16,-6,7,-6,-3,-4,-1,-16,0,-5,24,-2,-4,9,-1,2,-8,15,-6,4,11,-3,0,6,7,-10,2,-9,-7,-6,12,15,24,-1,-8,-9,15,-15,-3,-5,17,-10,11,13,-2,4,-15,-1,-2,-23,4,3,-16,-14,-7,-5,-3,-9,-10,3,-5,-1,-2,4,-1,8,1,9,12,-14,9,17,-9,0,-3,4,5,-6,13,-8,-1,10,19,-5,8,2,-15,-9,-12,-5,-4,0,12,4,24,-2,8,4,14,-4,8,16,-7,-1,5,-4,-8,18,-2,17,-5,-2,8,-2,-9,-7,3,-6,1,-22,-5,-2,-5,-10,-8,1,14,-13,-3,9,3,-1,-4,0,-1,-21,-7,-19,12,8,-8,8,24,-6,12,3,-2,-11,-5,-4,-22,5,-3,4,-4,24,-16,-9,7,23,-10,18,-9,12,1,21,17,-6,24,-11,-3,17,-7,-6,1,4,4,-7,2,6,14,3,-12,0,-6,13,-16,5,-10,12,7,2,5,-3,6,0,7,1,-23,-5,15,14,1,-1,-3,6,6,-9,6,12,-9,-2,4,7,-4,-5,-4,4,4,0,-13,-10,6,-12,2,-3,-6,0,16,3,-3,-14,5,11,6,11,5,-13,0,5,7,-5,-1,4,12,10,6,4,-10,-11,-1,10,4,5,-14,-14,11,0,-13,8,2,24,12,3,-1,2,-1,-14,9,3,-23,-6,-8,9,0,14,-15,-10,10,-6,-10,-5,-7,5,11,-15,-3,0,1,8,1,-6,-11,-18,-4,0,9,-4,22,-1,-5,4,-9,2,-20,6,1,2,1,-12,-9,15,5,-6,4,4,19,11,4,-4,17,-1,-8,-12,-8,-3,7,9,11,1,8,22,9,15,-15,-7,-7,-23,1,13,-5,2,-8,-5,3,-11,11,-18,3,-5,14,7,-20,-23,-10,-5,-2,0,6,-13,-17,2,-3,-1,-6,-2,14,-16,-12,6,15,-2,-12,-19,3};

void BriefRIDescriptorExtractor::rotatePattern( float theta ){
    // cout << "yup, the pattern is being rotated to align the gravity" << endl;
    theta = theta / 180.0 * PI;
    size_t bits = bytes_ * 8;
    float cos_theta = cos( theta );
    float sin_theta = sin( theta );

    for ( size_t i = 0; i < bits; ++i ){
        size_t offset = (i<<2);
        float x1 = pattern_[ offset ];
        float y1 = pattern_[ offset + 1 ];
        float x2 = pattern_[ offset + 2 ];
        float y2 = pattern_[ offset + 3 ];
        int x1_new = x1*cos_theta - y1*sin_theta;
        if ( x1_new < -23 ){
            x1_new = -23;
        }else{
            if ( x1_new > 24 ){
                x1_new = 24;
            }
        }
        int y1_new = x1*sin_theta + y1*cos_theta;
        if ( y1_new < -23 ){
            y1_new = -23;
        }else{
            if ( y1_new > 24 ){
                y1_new = 24;
            }
        }
        int x2_new = x2*cos_theta - y2*sin_theta;
        if ( x2_new < -23 ){
            x2_new = -23;
        }else{
            if ( x2_new > 24 ){
                x2_new = 24;
            }
        }
        int y2_new = x2*sin_theta + y2*cos_theta;
        if ( y2_new < -23 ){
            y2_new = -23;
        }else{
            if ( y2_new > 24 ){
                y2_new = 24;
            }
        }
        pattern[ i ][ 0 ] = x1_new;
        pattern[ i ][ 1 ] = y1_new;
        pattern[ i ][ 2 ] = x2_new;
        pattern[ i ][ 3 ] = y2_new;
    }
}


inline int smoothedSum(const Mat& sum, const KeyPoint& pt, int y, int x)
{
    static const int HALF_KERNEL = BriefDescriptorExtractor::KERNEL_SIZE / 2;

    int img_y = (int)(pt.pt.y + 0.5) + y;
    int img_x = (int)(pt.pt.x + 0.5) + x;
    return   sum.at<int>(img_y + HALF_KERNEL + 1, img_x + HALF_KERNEL + 1)
           - sum.at<int>(img_y + HALF_KERNEL + 1, img_x - HALF_KERNEL)
           - sum.at<int>(img_y - HALF_KERNEL, img_x + HALF_KERNEL + 1)
           + sum.at<int>(img_y - HALF_KERNEL, img_x - HALF_KERNEL);
}

static void pixelTests16(const Mat& sum, const std::vector<KeyPoint>& keypoints, Mat& descriptors, const int pattern[512][4])
{
    for (int i = 0; i < (int)keypoints.size(); ++i)
    {
        uchar* desc = descriptors.ptr(i);
        const KeyPoint& pt = keypoints[i];
        for ( size_t j = 0; j < 16; ++j ){
            desc[ j ] = 0;
            for ( size_t k = 0; k < 8; ++k ){
                int offset = ( j << 3 ) + k;
                desc[ j ] = (desc[ j ] << 1) + 
                    ( smoothedSum( sum, pt, 
                                   pattern[ offset ][ 1 ], 
                                   pattern[ offset ][ 0 ] ) 
                    < smoothedSum( sum, pt, 
                                   pattern[ offset ][ 3 ], 
                                   pattern[ offset ][ 2 ] ) 
                    );
            }
        }
    }
}

static void pixelTests32(const Mat& sum, const std::vector<KeyPoint>& keypoints, Mat& descriptors, const int pattern[512][4]){
    for (int i = 0; i < (int)keypoints.size(); ++i){
        uchar* desc = descriptors.ptr(i);
        const KeyPoint& pt = keypoints[i];
        for ( size_t j = 0; j < 32; ++j ){
            desc[ j ] = 0;
            for ( size_t k = 0; k < 8; ++k ){
                int offset = ( j << 3 ) + k;
                desc[ j ] = (desc[ j ] << 1) + 
                    ( smoothedSum( sum, pt, 
                                   pattern[ offset ][ 1 ], 
                                   pattern[ offset ][ 0 ] ) 
                    < smoothedSum( sum, pt, 
                                   pattern[ offset ][ 3 ], 
                                   pattern[ offset ][ 2 ] ) 
                    );
            }
        }
    }
}

static void pixelTests64(const Mat& sum, const std::vector<KeyPoint>& keypoints, Mat& descriptors, const int pattern[512][4])
{
    for (int i = 0; i < (int)keypoints.size(); ++i)
    {
        uchar* desc = descriptors.ptr(i);
        const KeyPoint& pt = keypoints[i];
        for ( size_t j = 0; j < 64; ++j ){
            desc[ j ] = 0;
            for ( size_t k = 0; k < 8; ++k ){
                int offset = ( j << 3 ) + k;
                desc[ j ] = (desc[ j ] << 1) + 
                    ( smoothedSum( sum, pt, 
                                   pattern[ offset ][ 1 ], 
                                   pattern[ offset ][ 0 ] ) 
                    < smoothedSum( sum, pt, 
                                   pattern[ offset ][ 3 ], 
                                   pattern[ offset ][ 2 ] ) 
                    );
            }
        }
    }
}

BriefRIDescriptorExtractor::BriefRIDescriptorExtractor(int bytes) :
    BriefDescriptorExtractor( bytes ){
    switch (bytes)
    {
        case 16:
            test_pattern_fn_ = pixelTests16;
            break;
        case 32:
            test_pattern_fn_ = pixelTests32;
            break;
        case 64:
            test_pattern_fn_ = pixelTests64;
            break;
        default:
            CV_Error(CV_StsBadArg, "bytes must be 16, 32, or 64");
    }
}

// int BriefRIDescriptorExtractor::descriptorSize() const
// {
//     return bytes_;
// }
// 
// int BriefRIDescriptorExtractor::descriptorType() const
// {
//     return CV_8UC1;
// }

void BriefRIDescriptorExtractor::read( const FileNode& fn)
{
    int dSize = fn["descriptorSize"];
    switch (dSize)
    {
        case 16:
            test_pattern_fn_ = pixelTests16;
            break;
        case 32:
            test_pattern_fn_ = pixelTests32;
            break;
        case 64:
            test_pattern_fn_ = pixelTests64;
            break;
        default:
            CV_Error(CV_StsBadArg, "descriptorSize must be 16, 32, or 64");
    }
    bytes_ = dSize;
}

void BriefRIDescriptorExtractor::computeImpl(const Mat& image, std::vector<KeyPoint>& keypoints, Mat& descriptors) const
{
    // cout << "yup, descriptor is being computed" << endl;
    // Construct integral image for fast smoothing (box filter)
    Mat sum;

    Mat grayImage = image;
    if( image.type() != CV_8U ) cvtColor( image, grayImage, CV_BGR2GRAY );

    ///TODO allow the user to pass in a precomputed integral image
    //if(image.type() == CV_32S)
    //  sum = image;
    //else

    integral( grayImage, sum, CV_32S);

    //Remove keypoints very close to the border
    KeyPointsFilter::runByImageBorder(keypoints, image.size(), PATCH_SIZE/2 + KERNEL_SIZE/2);

    descriptors = Mat::zeros((int)keypoints.size(), bytes_, CV_8U);
    test_pattern_fn_(sum, keypoints, descriptors, pattern);
}

