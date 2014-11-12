//
//    AstDetector - the interface class for the AGAST corner detector
//
//    Copyright (C) 2010  Elmar Mair
//
//    This program is free software: you can redistribute it and/or modify
//    it under the terms of the GNU General Public License as published by
//    the Free Software Foundation, either version 3 of the License, or
//    (at your option) any later version.
//
//    This program is distributed in the hope that it will be useful,
//    but WITHOUT ANY WARRANTY; without even the implied warranty of
//    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//    GNU General Public License for more details.
//
//    You should have received a copy of the GNU General Public License
//    along with this program.  If not, see <http://www.gnu.org/licenses/>.

#include "agastpp/AstDetector.h"
#include "agastpp/cvWrapper.h"

namespace agastpp {

using namespace std;
void AstDetector::score(const unsigned char* im,
                        const std::vector<CvPoint>& corners_all) {
  size_t num_corners = corners_all.size();

  if (num_corners > scores.capacity()) {
    if (scores.capacity() == 0) {
      scores.reserve(512 > num_corners ? 512 : num_corners);
    } else {
      size_t scores_cap_double = scores.capacity() * 2;
      if (num_corners > scores_cap_double) scores_cap_double = num_corners;
      scores.reserve(scores_cap_double);
    }
  }

  scores.resize(num_corners);

  for (size_t n = 0; n < num_corners; ++n)
    scores[n] = cornerScore(im + corners_all[n].y * xsize + corners_all[n].x);
}

void AstDetector::nms(const unsigned char* im,
                      const std::vector<CvPoint>& corners_all,
                      std::vector<CvPoint>& corners_nms) {
  score(im, corners_all);
  nonMaximumSuppression(corners_all, corners_nms);
}

}  // namespace agastpp
