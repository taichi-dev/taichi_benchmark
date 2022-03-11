#pragma once

//-----------------------------------------------------------------------------
// Includes
//-----------------------------------------------------------------------------
#pragma region

#include "vector.hpp"

#pragma endregion

//-----------------------------------------------------------------------------
// System Includes
//-----------------------------------------------------------------------------
#pragma region

#include <cstdio>

#pragma endregion

#define fopen_s(pFile,filename,mode) ((*(pFile))=fopen((filename),(mode)))==NULL

//-----------------------------------------------------------------------------
// Declarations and Definitions
//-----------------------------------------------------------------------------
namespace smallpt {

	inline void WritePPM(std::uint32_t w, 
						 std::uint32_t h, 
						 const Vector3* Ls, 
						 const char* fname = "cpp-image.ppm") noexcept {
		
		FILE* fp;
		
		fopen_s(&fp, fname, "w");
		
		std::fprintf(fp, "P3\n%u %u\n%u\n", w, h, 255u);
		for (std::size_t i = 0; i < w * h; ++i) {
			std::fprintf(fp, "%u %u %u ", 
						 ToByte(Ls[i].m_x), 
						 ToByte(Ls[i].m_y), 
						 ToByte(Ls[i].m_z));
		}
		
		std::fclose(fp);
	}
}
