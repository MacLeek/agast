//
//    demo - demonstration code to show the usage of AGAST, an adaptive and
//           generic corner detector based on the accelerated segment test
//
//    Copyright (C) 2010  Elmar Mair
//    All rights reserved.
//
//    Redistribution and use in source and binary forms, with or without
//    modification, are permitted provided that the following conditions are
//    met:
//        * Redistributions of source code must retain the above copyright
//          notice, this list of conditions and the following disclaimer.
//        * Redistributions in binary form must reproduce the above copyright
//          notice, this list of conditions and the following disclaimer in the
//          documentation and/or other materials provided with the distribution.
//        * Neither the name of the <organization> nor the
//          names of its contributors may be used to endorse or promote products
//          derived from this software without specific prior written
//          permission.
//
//    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
//    "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
//    TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
//    PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR
//    ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
//    DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
//    SERVICES;
//    LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSES
//    AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR
//    TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
//    OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#include <iostream>
#include <cstdio>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "agastpp/agast5_8.h"
#include "agastpp/agast7_12s.h"
#include "agastpp/agast7_12d.h"
#include "agastpp/oast9_16.h"

using namespace std;

// threshold for accelerated segment test (16-, 12- and 8-pixel mask)
#define AST_THR_16 40
#define AST_THR_12 38
#define AST_THR_8 27

enum AST_PATTERN {
  OAST9_16,
  AGAST7_12d,
  AGAST7_12s,
  AGAST5_8,
  AST_PATTERN_LENGTH
};

static void drawResult(const cv::Mat &imageGray, cv::Mat &imageOut,
                       const vector<CvPoint> corners_all,
                       const vector<CvPoint> corners_nms) {
  cv::cvtColor(imageGray, imageOut, CV_GRAY2RGB);
  for (unsigned int i = 0; i < corners_all.size(); i++) {
    cv::line(imageOut, cv::Point2f(corners_all[i].x, corners_all[i].y),
             cv::Point2f(corners_all[i].x, corners_all[i].y),
             cv::Scalar(255, 0, 0));
  }
  for (unsigned int i = 0; i < corners_nms.size(); i++) {
    // points
    cv::line(imageOut, cv::Point2f(corners_nms[i].x, corners_nms[i].y),
             cv::Point2f(corners_nms[i].x, corners_nms[i].y),
             cv::Scalar(0, 255, 0));
    // crosses
    //		cvLine( imageOut, cvPoint( corners_nms[i].x-1, corners_nms[i].y
    //),
    // cvPoint( corners_nms[i].x+1, corners_nms[i].y ), CV_RGB(0,255,0) );
    //		cvLine( imageOut,	cvPoint( corners_nms[i].x,
    //corners_nms[i].y-1
    //),
    // cvPoint( corners_nms[i].x, corners_nms[i].y+1 ), CV_RGB(0,255,0) );
  }
}

int main(int argc, char *argv[]) {
  char *name_imageIn;
  cv::Mat imageIn, imageGray, imageOut;
  int rows, cols;

  // check program parameters
  if (argc != 2) {
    printf(
        "Wrong number of arguments - need 1 argument:\ndemo "
        "<image_in.xxx>\ne.g. demo demo.ppm\n");
    exit(0);
  }
  name_imageIn = argv[1];

  cout << "Starting demo...\n";

  // load image and convert it to 8 bit grayscale
  imageIn = cv::imread(name_imageIn, -1);
  if (imageIn.rows == 0) {
    cout << "Image \"" << name_imageIn << "\" could not be loaded.\n";
    exit(0);
  }
  imageGray = cv::Mat(imageIn.size(), IPL_DEPTH_8U, 1);
  cv::cvtColor(imageIn, imageGray, CV_RGB2GRAY);

  cols = imageGray.cols;
  rows = imageGray.rows;

  imageOut = cv::Mat(imageIn.size(), IPL_DEPTH_8U, 3);

  for (int j = 0; j < AST_PATTERN_LENGTH; j++) {
    cv::cvtColor(imageGray, imageOut, CV_GRAY2RGB);
    switch (j) {
      case OAST9_16: {
        cout << "OAST9_16:   ";
        OastDetector9_16 ad9_16(cols, rows, AST_THR_16);
        ad9_16.processImage((unsigned char *)imageGray.data);
        vector<CvPoint> corners_all = ad9_16.get_corners_all();
        vector<CvPoint> corners_nms = ad9_16.get_corners_nms();

        cout << corners_all.size() << " corner responses - "
             << corners_nms.size() << " corners after non-maximum suppression."
             << endl;
        drawResult(imageGray, imageOut, corners_all, corners_nms);
        cv::imwrite("oast9_16.ppm", imageOut);
        break;
      }
      case AGAST7_12d: {
        cout << "AGAST7_12d: ";
        AgastDetector7_12d ad7_12d(cols, rows, AST_THR_12);
        ad7_12d.processImage((unsigned char *)imageGray.data);
        vector<CvPoint> corners_all = ad7_12d.get_corners_all();
        vector<CvPoint> corners_nms = ad7_12d.get_corners_nms();

        cout << corners_all.size() << " corner responses - "
             << corners_nms.size() << " corners after non-maximum suppression."
             << endl;
        drawResult(imageGray, imageOut, corners_all, corners_nms);
        cv::imwrite("agast7_12d.ppm", imageOut);
        break;
      }
      case AGAST7_12s: {
        cout << "AGAST7_12s: ";
        AgastDetector7_12s ad7_12s(cols, rows, AST_THR_12);
        ad7_12s.processImage((unsigned char *)imageGray.data);
        vector<CvPoint> corners_all = ad7_12s.get_corners_all();
        vector<CvPoint> corners_nms = ad7_12s.get_corners_nms();

        cout << corners_all.size() << " corner responses - "
             << corners_nms.size() << " corners after non-maximum suppression."
             << endl;
        drawResult(imageGray, imageOut, corners_all, corners_nms);
        cv::imwrite("agast7_12s.ppm", imageOut);
        break;
      }
      case AGAST5_8: {
        cout << "AGAST5_8:   ";
        AgastDetector5_8 ad5_8(cols, rows, AST_THR_8);
        ad5_8.processImage((unsigned char *)imageGray.data);
        vector<CvPoint> corners_all = ad5_8.get_corners_all();
        vector<CvPoint> corners_nms = ad5_8.get_corners_nms();

        cout << corners_all.size() << " corner responses - "
             << corners_nms.size() << " corners after non-maximum suppression."
             << endl;
        drawResult(imageGray, imageOut, corners_all, corners_nms);
        cv::imwrite("agast5_8.ppm", imageOut);

        // parallel image processing by two detectors (possible threads)
        AgastDetector5_8 ad5_8_thread1;  // necessary to access
                                         // get_borderWidth() in VisualStudio
        ad5_8_thread1 =
            AgastDetector5_8(cols, ceil(static_cast<float>(rows) * 0.5) +
                                       ad5_8_thread1.get_borderWidth(),
                             AST_THR_8);
        AgastDetector5_8 ad5_8_thread2(cols,
                                       floor(static_cast<float>(rows) * 0.5) +
                                           ad5_8_thread1.get_borderWidth(),
                                       AST_THR_8);
        ad5_8_thread1.processImage((unsigned char *)imageGray.data);
        ad5_8_thread2.processImage(
            ((unsigned char *)imageGray.data) +
            ((int)ceil((float)rows * 0.5) - ad5_8_thread2.get_borderWidth()) *
                cols);
        vector<CvPoint> corners_all_thread1 = ad5_8_thread1.get_corners_all();
        vector<CvPoint> corners_all_thread2 = ad5_8_thread2.get_corners_all();
        vector<CvPoint> corners_nms_thread1 = ad5_8_thread1.get_corners_nms();
        vector<CvPoint> corners_nms_thread2 = ad5_8_thread2.get_corners_nms();

        // adjust thread2 responses from ROI to whole image scope
        int offset =
            ((int)ceil((float)rows * 0.5) - ad5_8_thread2.get_borderWidth());
        for (vector<CvPoint>::iterator i = corners_all_thread2.begin();
             i < corners_all_thread2.end(); i++)
          i->y += offset;
        for (vector<CvPoint>::iterator i = corners_nms_thread2.begin();
             i < corners_nms_thread2.end(); i++)
          i->y += offset;

        cout << "  thread 1:   " << corners_all_thread1.size()
             << " corner responses - " << corners_nms_thread1.size()
             << " corners after non-maximum suppression." << endl;
        cout << "  thread 2:   " << corners_all_thread2.size()
             << " corner responses - " << corners_nms_thread2.size()
             << " corners after non-maximum suppression." << endl;
        cout << "  thread 1+2: "
             << corners_all_thread1.size() + corners_all_thread2.size()
             << " corner responses - "
             << corners_nms_thread1.size() + corners_nms_thread2.size()
             << " corners after non-maximum suppression." << endl;
        drawResult(imageGray, imageOut, corners_all_thread1,
                   corners_nms_thread1);
        cv::imwrite("agast5_8_thread1.ppm", imageOut);
        drawResult(imageGray, imageOut, corners_all_thread2,
                   corners_nms_thread2);
        cv::imwrite("agast5_8_thread2.ppm", imageOut);
        break;
      }
      default:
        break;
    }
  }

  cout << "...done!\n";

  return 0;
}
