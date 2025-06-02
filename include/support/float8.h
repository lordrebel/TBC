

#pragma once

#include <stdint.h>

namespace tbc {

float get_f8e4m3_max();

float get_f8e4m3_min();

float get_f8e5m2_max();

float get_f8e5m2_min();

/*
convert f32 to f8e4m3 f8e5m2 by uint8
*/
uint8_t f32_to_f8e4m3(float src);
uint8_t f32_to_f8e5m2(float src);
float f8e4m3_to_f32(uint8_t src);
float f8e5m2_to_f32(uint8_t src);

/*
convert f16 to f8e4m3 f8e5m2 by uint8
*/
uint8_t f16_to_f8e4m3(float src);
uint8_t f16_to_f8e5m2(float src);
float f8e4m3_to_f16(uint8_t src);
float f8e5m2_to_f16(uint8_t src);

/*
  convert f32 to f8r4m3 and back to f32
*/
float F8E4M3(float src);
float F8E5M2(float src);
void F8E4M3(const float *p_src, float *p_dst, int num, float step);
void F8E5M2(const float *p_src, float *p_dst, int num, float step);

} // namespace tbc
