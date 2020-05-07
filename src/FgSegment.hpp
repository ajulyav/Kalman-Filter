/*
 * FgSegment.hpp
 */

#ifndef SRC_FGSEGMENTATION_HPP_
#define SRC_FGSEGMENTATION_HPP_

#include "opencv2/opencv.hpp"
#include <cstring>
#include <iostream>

using namespace cv;
using namespace std;

// Params for blob extraction

/* For abandonedBox_600_1000_clip
 *
 #define MIN_WIDTH 20
 #define MIN_HEIGHT 20
 *
*/

/* For boats_6950_7900_clip
 *
 #define MIN_WIDTH 30
 #define MIN_HEIGHT 80
 *
 */


/* For pedestrians_800_1025_clip
 *
 #define MIN_WIDTH 20
 #define MIN_HEIGHT 20
 *
 */

/* For streetCornerAtNight_0_100_clip
 *
 #define MIN_WIDTH 80
 #define MIN_HEIGHT 80
 *
 */

#define MIN_WIDTH 20
#define MIN_HEIGHT 20


// Maximun number of char in the blob's format
const int MAX_FORMAT = 1024;

typedef enum {
	UNKNOWN=0
} CLASS;

struct cvBlob {
	int     ID;  //blob ID
	int   x, y;  //position
	int   w, h;  //blob weight and height
	CLASS label; //type - Unknown
	int center_x,center_y;
	int area ;
	char format[MAX_FORMAT];

};

inline cvBlob initBlob(int id, int x, int y, int w, int h, int center_x , int center_y, int area)
{
	cvBlob B = { id, x, y, w, h, UNKNOWN, center_x, center_y, area};
	return B;
}


class FgSegment{

    private:

	   std::vector<cvBlob> bloblist;
       std::vector<Point> blob_centers;

       bool blob_extracted = false;

	public:
		Mat bkg_subtraction(Mat frame);
		Mat MorphologicalOpen(Mat fgmask);
		void extractBlobs(Mat fgmask);
        Mat paintBlobImage(Mat frame, std::vector<cvBlob> bloblist);


//GETTERS AND SETTERS
    const std::vector<cvBlob>& getBloblist() const {return bloblist;}
	const std::vector<Point>& getBlobCenters() const{return blob_centers;}
	bool BlobExists() const{return blob_extracted;}

};

#endif
