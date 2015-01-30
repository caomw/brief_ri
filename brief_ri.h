#ifndef __BRIEF_RI_H
#define __BRIEF_RI_H

#include <opencv2/opencv.hpp>

class BriefRIDescriptorExtractor: public cv::BriefDescriptorExtractor{
    BriefRIDescriptorExtractor( int bytes )
    : cv::BriefDescriptorExtractor( bytes ) {
    }
}

#endif
