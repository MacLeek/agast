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
    cv::circle(imageOut, cv::Point2f(corners_all[i].x, corners_all[i].y), 1,
               cv::Scalar(255, 0, 0), -1, CV_AA);
  }

  for (unsigned int i = 0; i < corners_nms.size(); i++) {
    cv::circle(imageOut, cv::Point2f(corners_nms[i].x, corners_nms[i].y), 1,
               cv::Scalar(0, 255, 0), -1, CV_AA);
  }
}

int main(int argc, char **argv) {
  string name_image_in;
  cv::Mat image_gray, image_rgb;
  int rows, cols;

  // check program parameters
  if (argc != 2) {
    printf(
        "Wrong number of arguments - need 1 argument:\ndemo "
        "<image_in.xxx>\ne.g. demo demo.ppm\n");
    exit(0);
  }
  name_image_in = argv[1];

  cout << "Starting demo...\n";

  // load image and convert it to 8 bit grayscale
  image_gray = cv::imread(name_image_in, CV_LOAD_IMAGE_GRAYSCALE);
  if (image_gray.empty()) {
    cout << "Image \"" << name_image_in << "\" could not be loaded.\n";
    exit(0);
  }

  cols = image_gray.cols;
  rows = image_gray.rows;

  cv::cvtColor(image_gray, image_rgb, CV_GRAY2BGR);

  for (int j = 0; j < AST_PATTERN_LENGTH; j++) {
    cv::cvtColor(image_gray, image_rgb, CV_GRAY2RGB);
    switch (j) {
      case OAST9_16: {
        cout << "OAST9_16:   ";
        agastpp::OastDetector9_16 ad9_16(cols, rows, AST_THR_16);
        vector<CvPoint> corners_all;
        vector<CvPoint> corners_nms;
        ad9_16.processImage((unsigned char *)image_gray.data, corners_all,
                            corners_nms);

        cout << corners_all.size() << " corner responses - "
             << corners_nms.size() << " corners after non-maximum suppression."
             << endl;
        drawResult(image_gray, image_rgb, corners_all, corners_nms);
        cv::imwrite("oast9_16.ppm", image_rgb);
        break;
      }
      case AGAST7_12d: {
        cout << "AGAST7_12d: ";
        agastpp::AgastDetector7_12d ad7_12d(cols, rows, AST_THR_12);
        vector<CvPoint> corners_all;
        vector<CvPoint> corners_nms;
        ad7_12d.processImage((unsigned char *)image_gray.data, corners_all,
                             corners_nms);

        cout << corners_all.size() << " corner responses - "
             << corners_nms.size() << " corners after non-maximum suppression."
             << endl;
        drawResult(image_gray, image_rgb, corners_all, corners_nms);
        cv::imwrite("agast7_12d.ppm", image_rgb);
        break;
      }
      case AGAST7_12s: {
        cout << "AGAST7_12s: ";
        agastpp::AgastDetector7_12s ad7_12s(cols, rows, AST_THR_12);
        vector<CvPoint> corners_all;
        vector<CvPoint> corners_nms;
        ad7_12s.processImage((unsigned char *)image_gray.data, corners_all,
                             corners_nms);

        cout << corners_all.size() << " corner responses - "
             << corners_nms.size() << " corners after non-maximum suppression."
             << endl;
        drawResult(image_gray, image_rgb, corners_all, corners_nms);
        cv::imwrite("agast7_12s.ppm", image_rgb);
        break;
      }
      case AGAST5_8: {
        cout << "AGAST5_8:   ";
        agastpp::AgastDetector5_8 ad5_8(cols, rows, AST_THR_8);
        vector<CvPoint> corners_all;
        vector<CvPoint> corners_nms;
        ad5_8.processImage((unsigned char *)image_gray.data, corners_all,
                           corners_nms);

        cout << corners_all.size() << " corner responses - "
             << corners_nms.size() << " corners after non-maximum suppression."
             << endl;
        drawResult(image_gray, image_rgb, corners_all, corners_nms);
        cv::imwrite("agast5_8.ppm", image_rgb);

        // parallel image processing by two detectors (possible threads)
        agastpp::AgastDetector5_8 ad5_8_thread1;  // necessary to access
        // get_borderWidth() in VisualStudio
        ad5_8_thread1 = agastpp::AgastDetector5_8(
            cols, ceil(static_cast<float>(rows) * 0.5) +
                      ad5_8_thread1.get_borderWidth(),
            AST_THR_8);
        agastpp::AgastDetector5_8 ad5_8_thread2(
            cols, floor(static_cast<float>(rows) * 0.5) +
                      ad5_8_thread1.get_borderWidth(),
            AST_THR_8);
        vector<CvPoint> corners_all_thread1;
        vector<CvPoint> corners_all_thread2;
        vector<CvPoint> corners_nms_thread1;
        vector<CvPoint> corners_nms_thread2;
        ad5_8_thread1.processImage(image_gray.data, corners_all_thread1,
                                   corners_nms_thread1);
        ad5_8_thread2.processImage(
            (image_gray.data) + ((int)ceil((float)rows * 0.5) -
                                 ad5_8_thread2.get_borderWidth()) *
                                    cols,
            corners_all_thread2, corners_nms_thread2);

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
        drawResult(image_gray, image_rgb, corners_all_thread1,
                   corners_nms_thread1);
        cv::imwrite("agast5_8_thread1.ppm", image_rgb);
        drawResult(image_gray, image_rgb, corners_all_thread2,
                   corners_nms_thread2);
        cv::imwrite("agast5_8_thread2.ppm", image_rgb);
        break;
      }
      default:
        break;
    }
  }

  cout << "...done!\n";

  return 0;
}
