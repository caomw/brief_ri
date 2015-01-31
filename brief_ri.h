#ifndef __BRIEF_RI_H
#define __BRIEF_RI_H

#include <opencv2/opencv.hpp>

class BriefRIDescriptorExtractor: public cv::BriefDescriptorExtractor{
public:
    BriefRIDescriptorExtractor( int bytes = 32 );
    void rotatePattern( float theta );
    virtual void read( const cv::FileNode& );
    // virtual void write( FileStorage& ) const;

    // virtual int descriptorSize() const;
    // virtual int descriptorType() const;
protected:
    virtual void computeImpl(const cv::Mat& image, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors) const;
    int pattern[512][4];
    typedef void(*PixelTestWithPatternFn)(const cv::Mat&, const std::vector<cv::KeyPoint>&, cv::Mat&, const int pattern[512][4]);
    PixelTestWithPatternFn test_pattern_fn_;

};

#endif
