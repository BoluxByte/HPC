#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "image.h"
#include "sobel.h"

#define GAUSSIAN_KERNEL_SIZE    3
#define SOBEL_KERNEL_SIZE       3
#define SOBEL_BINARY_THRESHOLD  150  // in the range 0 to uint8_max (255)
#define CONVOLUTION_PIXEL(img, kernel, ponderation, i) \
    (img->data[i - img->width - 1] * kernel[0] + \
         img->data[i - img->width] * kernel[1] + \
         img->data[i - img->width + 1] * kernel[2] + \
         img->data[i - 1] * kernel[3] + \
         img->data[i] * kernel[4] + \
         img->data[i + 1] * kernel[5] + \
         img->data[i + img->width - 1] * kernel[6] + \
         img->data[i + img->width] * kernel[7] + \
         img->data[i + img->width + 1] * kernel[8]) / ponderation

#define CONVOLUTION_MATRIX(matrix, kernel, ponderation) \
        (matrix[0][0] * kernel[0] + \
         matrix[1][0] * kernel[1] + \
         matrix[2][0] * kernel[2] + \
         matrix[3][0] * kernel[3] + \
         matrix[4][0] * kernel[4] + \
         matrix[5][0] * kernel[5] + \
         matrix[6][0] * kernel[6] + \
         matrix[7][0] * kernel[7] + \
         matrix[8][0] * kernel[8]) / ponderation


const int16_t sobel_v_kernel[SOBEL_KERNEL_SIZE*SOBEL_KERNEL_SIZE] = {
    -1, -2, -1,
     0,  0,  0,
     1,  2,  1,
};

const int16_t sobel_h_kernel[SOBEL_KERNEL_SIZE*SOBEL_KERNEL_SIZE] = {
    -1,  0,  1,
    -2,  0,  2,
    -1,  0,  1,
};

const uint16_t gauss_kernel[GAUSSIAN_KERNEL_SIZE*GAUSSIAN_KERNEL_SIZE] = {
    1, 2, 1,
    2, 4, 2,
    1, 2, 1,
};

struct img_1D_t *edge_detection_1D(const struct img_1D_t *input_img){
    struct img_1D_t *a_img =  allocate_image_1D(input_img->width, input_img->height, COMPONENT_GRAYSCALE);
    struct img_1D_t *b_img = allocate_image_1D(input_img->width, input_img->height, COMPONENT_GRAYSCALE);
    rgb_to_grayscale_1D(input_img, a_img);
    gaussian_filter_1D(a_img, b_img, gauss_kernel);
    sobel_filter_1D(b_img, a_img, sobel_v_kernel, sobel_h_kernel);
    free_image(b_img);
    return a_img;
    }

void rgb_to_grayscale_1D(const struct img_1D_t *img, struct img_1D_t *result){
    result->components = COMPONENT_GRAYSCALE;
    result->width = img->width;
    result->height = img->height;
    for (size_t i = 0; i < img->width * img->height; i++) {
        result->data[i] = (uint8_t)(img->data[i * img->components + R_OFFSET] * FACTOR_R +
                                    img->data[i * img->components + G_OFFSET] * FACTOR_G +
                                    img->data[i * img->components + B_OFFSET] * FACTOR_B);
    }
}

void gaussian_filter_1D(const struct img_1D_t *img, struct img_1D_t *res_img, const uint16_t *kernel){
    const uint16_t gauss_ponderation = 16;
    for (size_t i = 0; i < img->width * img->height; i++) {
        //if on the border, we keep the pixel value
        if(i < img->width || i % img->width == 0 || i % img->width == img->width - 1 || i > img->width * img->height - img->width){
            res_img->data[i] = img->data[i];
            continue;
        }
        res_img->data[i] = CONVOLUTION_PIXEL(img, kernel, gauss_ponderation, i);

    }
}

void sobel_filter_1D(const struct img_1D_t *img, struct img_1D_t *res_img, const int16_t *v_kernel, const int16_t *h_kernel) {
    for (size_t i = 0; i < img->width * img->height; i++) {
        if (i < img->width || i % img->width == 0 || i % img->width == img->width - 1 ||
            i > img->width * img->height - img->width) {
            res_img->data[i] = img->data[i];
        } else {
            int16_t gx = CONVOLUTION_PIXEL(img, h_kernel, 1, i);

            int16_t gy = CONVOLUTION_PIXEL(img, v_kernel, 1, i);

            res_img->data[i] = (sqrt(gx * gx + gy * gy) > SOBEL_BINARY_THRESHOLD) ? GRAYSCALE_WHITE : GRAYSCALE_BLACK;

        }
    }
}

struct img_chained_t *edge_detection_chained(const struct img_chained_t *input_img){
    struct img_chained_t *a_img = allocate_image_chained(input_img->width, input_img->height, COMPONENT_GRAYSCALE);
    struct img_chained_t *b_img = allocate_image_chained(input_img->width, input_img->height, COMPONENT_GRAYSCALE);
    rgb_to_grayscale_chained(input_img, a_img);
    gaussian_filter_chained(a_img, b_img, gauss_kernel);
    sobel_filter_chained(b_img, a_img, sobel_v_kernel, sobel_h_kernel);
    return a_img;
}



void rgb_to_grayscale_chained(const struct img_chained_t *img, struct img_chained_t *result){
    result->components = COMPONENT_GRAYSCALE;
    result->width = img->width;
    result->height = img->height;
    struct pixel_t *current_pixel = img->first_pixel;
    struct pixel_t *current_result_pixel = result->first_pixel;
    while(current_pixel != NULL){
        current_result_pixel->pixel_val[0] = (uint8_t)(current_pixel->pixel_val[R_OFFSET] * FACTOR_R +
                                    current_pixel->pixel_val[G_OFFSET] * FACTOR_G +
                                    current_pixel->pixel_val[B_OFFSET] * FACTOR_B);
        current_pixel = current_pixel->next_pixel;
        current_result_pixel = current_result_pixel->next_pixel;
    }
}

void gaussian_filter_chained(const struct img_chained_t *img, struct img_chained_t *res_img,const uint16_t *kernel){
    const uint16_t gauss_ponderation = 16;
    struct pixel_t *current_pixel = img->first_pixel;
    struct pixel_t *current_result_pixel = res_img->first_pixel;
    struct pixel_t *first_lane = img->first_pixel;
    struct pixel_t *second_lane = NULL;
    // on initialise la première partie de l'image
    for(int i = 0; i < img->width * 2; ++i){
        if (i == img->width){
            second_lane = current_pixel;
        }
        if (i < img->width ){
            current_result_pixel->pixel_val[0] = current_pixel->pixel_val[0];
            current_result_pixel = current_result_pixel->next_pixel;
        }
        current_pixel = current_pixel->next_pixel;
    }
    // on initialise le milieu de l'image
    for(int j = 0; j < img->height - 2; j++){
        current_result_pixel->pixel_val[0] = second_lane->pixel_val[0];
        current_result_pixel = current_result_pixel->next_pixel;
        for(int i = 0; i < img->width - 2; i++){
            uint8_t *pointer_matrix[9] = {first_lane->pixel_val, first_lane->next_pixel->pixel_val, first_lane->next_pixel->next_pixel->pixel_val,
                                                 second_lane->pixel_val, second_lane->next_pixel->pixel_val, second_lane->next_pixel->next_pixel->pixel_val,
                                                 current_pixel->pixel_val, current_pixel->next_pixel->pixel_val, current_pixel->next_pixel->next_pixel->pixel_val};
            current_result_pixel->pixel_val[0] = CONVOLUTION_MATRIX(pointer_matrix, kernel, gauss_ponderation);
            current_pixel = current_pixel->next_pixel;
            first_lane = first_lane->next_pixel;
            second_lane = second_lane->next_pixel;
            current_result_pixel = current_result_pixel->next_pixel;
        }
		second_lane = second_lane->next_pixel;
        current_result_pixel->pixel_val[0] = second_lane->pixel_val[0];
        current_pixel = current_pixel->next_pixel->next_pixel;
        current_result_pixel = current_result_pixel->next_pixel;
        first_lane = first_lane->next_pixel->next_pixel;
        second_lane = second_lane->next_pixel;
    }
    // on initialise la dernière partie de l'image
    for(int i = 0; i < img->width; i++){
        current_result_pixel->pixel_val[0] = second_lane->pixel_val[0];
		second_lane = second_lane->next_pixel;
        current_result_pixel = current_result_pixel->next_pixel;
	}

}

void sobel_filter_chained(const struct img_chained_t *img, struct img_chained_t *res_img, const int16_t *v_kernel, const int16_t *h_kernel){
	struct pixel_t *current_pixel = img->first_pixel;
    struct pixel_t *current_result_pixel = res_img->first_pixel;
    struct pixel_t *first_lane = img->first_pixel;
    struct pixel_t *second_lane = NULL;
    // on initialise la première partie de l'image
    for(int i = 0; i < img->width * 2; ++i){
        if (i == img->width){
            second_lane = current_pixel;
        }
        if (i < img->width ){
            current_result_pixel->pixel_val[0] = current_pixel->pixel_val[0];
            current_result_pixel = current_result_pixel->next_pixel;
        }
        current_pixel = current_pixel->next_pixel;
    }
    // on initialise le milieu de l'image
    for(int j = 0; j < img->height - 2; j++){
        current_result_pixel->pixel_val[0] = second_lane->pixel_val[0];
        current_result_pixel = current_result_pixel->next_pixel;
        for(int i = 0; i < img->width - 2; i++){
            uint8_t *pointer_matrix[9] = {first_lane->pixel_val, first_lane->next_pixel->pixel_val, first_lane->next_pixel->next_pixel->pixel_val,
                                                 second_lane->pixel_val, second_lane->next_pixel->pixel_val, second_lane->next_pixel->next_pixel->pixel_val,
                                                 current_pixel->pixel_val, current_pixel->next_pixel->pixel_val, current_pixel->next_pixel->next_pixel->pixel_val};
            int16_t gx = CONVOLUTION_MATRIX(pointer_matrix, v_kernel, 1);
            int16_t gy = CONVOLUTION_MATRIX(pointer_matrix, h_kernel, 1);
			current_result_pixel->pixel_val[0] = (sqrt(gx * gx + gy * gy) > SOBEL_BINARY_THRESHOLD) ? GRAYSCALE_WHITE : GRAYSCALE_BLACK;
            current_pixel = current_pixel->next_pixel;
            first_lane = first_lane->next_pixel;
            second_lane = second_lane->next_pixel;
            current_result_pixel = current_result_pixel->next_pixel;
        }
		second_lane = second_lane->next_pixel;
        current_result_pixel->pixel_val[0] = second_lane->pixel_val[0];
        current_pixel = current_pixel->next_pixel->next_pixel;
        current_result_pixel = current_result_pixel->next_pixel;
        first_lane = first_lane->next_pixel->next_pixel;
        second_lane = second_lane->next_pixel;
    }
    // on initialise la dernière partie de l'image
    for(int i = 0; i < img->width; i++){
        current_result_pixel->pixel_val[0] = second_lane->pixel_val[0];
		second_lane = second_lane->next_pixel;
        current_result_pixel = current_result_pixel->next_pixel;
	}
}

