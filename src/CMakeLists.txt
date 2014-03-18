cmake_minimum_required(VERSION 2.6)
if(DEFINED DEBUG)
	add_definitions(-DEBUG)
endif()
aux_source_directory(. SOURCES)
set(CUFILE CUDA.cu)
set(CFLAGS -O3 -Wall -Werror -std=gnu99)
set(LCPATH ${CMAKE_CURRENT_SOURCE_DIR}/locale/ru)
set(PO_FILE ${LCPATH}/messages.po)
set(MO_FILE ${LCPATH}/LC_MESSAGES/${PROJ}.mo)
set(RU_FILE ${LCPATH}/ru.po)
find_package(PkgConfig REQUIRED)
#include(FindOpenMP)
find_package(OpenMP)
find_package(JPEG)
find_package(TIFF)
find_package(PNG)
#find_package(OpenGL REQUIRED)
#find_package(GTK2 REQUIRED)
pkg_check_modules(${PROJ} REQUIRED
#	gtkglext-1.0>=0.7.0
#	gtkglext-x11-1.0>=0.7.0
	cfitsio>=3.0
	json>=0.9
#	libpng>=1.5
#	fftw3>=3.2.0
	)
if(OPENMP_FOUND)
	add_definitions(-DOMP)
	set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
	set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()
list(APPEND ${PROJ}_INCLUDE_DIRS ${JPEG_INCLUDE_DIR} ${PNG_INCLUDE_DIR} ${TIFF_INCLUDE_DIR})
list(APPEND ${PROJ}_LIBRARIES ${JPEG_LIBRARY} ${PNG_LIBRARY} ${TIFF_LIBRARY})
add_definitions(-D_XOPEN_SOURCE=1000  -D__JPEG=${JPEG_FOUND} -D__TIFF=${TIFF_FOUND} -D__PNG=${PNG_FOUND})
#list(APPEND ${${PROJ}_LIBRARY_DIRS} )
#set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -lfftw3_threads")
#if(NOT DEFINED NO_LEPTONICA)
#	pkg_check_modules(LIBLEPT liblept)
#endif()
#if(NOT DEFINED NO_GSL)
#	pkg_check_modules(GSL gsl)
#endif()
#if(NOT DEFINED GSL_VERSION)
#	message("GSL not found, some mathematics functions wouldn't be avialable")
#else()
#	add_definitions(-DGSL_FOUND)
#	list(APPEND ${${PROJ}_LIBRARIES} ${GSL_LIBRARIES})
#	list(APPEND ${${PROJ}_INCLUDE_DIRS} ${GSL_INCLUDE_DIRS})
#	list(APPEND ${${PROJ}_LIBRARY_DIRS} ${GSL_LIBRARY_DIRS})
#endif()
#if(NOT DEFINED LIBLEPT_VERSION)
#	message("Leptonica library not found, some functions wouldn't be avialable")
#else()
#	add_definitions(-DLEPTONICA_FOUND)
#	list(APPEND ${${PROJ}_LIBRARIES} ${LIBLEPT_LIBRARIES})
#	list(APPEND ${${PROJ}_INCLUDE_DIRS} ${LIBLEPT_INCLUDE_DIRS})
#	list(APPEND ${${PROJ}_LIBRARY_DIRS} ${LIBLEPT_LIBRARY_DIRS})
#endif()

include_directories(include ${${PROJ}_INCLUDE_DIRS})
link_directories(${${PROJ}_LIBRARY_DIRS})
add_definitions(-DPACKAGE_VERSION=\"${PROJ_VER}\"  -DGETTEXT_PACKAGE=\"${PROJ}\"
				-DLOCALEDIR=\"${LOCALEDIR}\" ${CFLAGS})

if(CUDA_FOUND)
	set(TEST ${CMAKE_BINARY_DIR}/test)
	set(TESTSRC ${CMAKE_CURRENT_SOURCE_DIR}/test/capability.cu)
	execute_process(COMMAND nvcc -lcuda ${TESTSRC} -o ${TEST})
	execute_process(COMMAND ${TEST} OUTPUT_VARIABLE CUDA_ARCH)
	message("Cuda architecture:  ${CUDA_ARCH}")
	list(APPEND CUDA_NVCC_FLAGS --use_fast_math ${CUDA_ARCH})
	#-gencode arch=compute_20,code=sm_20 -gencode arch=compute_12,code=sm_12 -gencode arch=compute_13,code=sm_13)
	cuda_include_directories(include)
	cuda_add_executable(../${PROJ} ${SOURCES} ${CUFILE} ${PO_FILE} ${MO_FILE})
	target_link_libraries(	../${PROJ} ${${PROJ}_LIBRARIES} ${CUDA_curand_LIBRARY}
							#${CUDA_CUFFT_LIBRARIES}
							)
else(CUDA_FOUND)
	add_executable(../${PROJ} ${SOURCES} ${PO_FILE} ${MO_FILE})
	target_link_libraries(	../${PROJ} ${${PROJ}_LIBRARIES}
							${CMAKE_THREAD_LIBS_INIT}
							)
endif(CUDA_FOUND)

# Installation of the program
INSTALL(FILES ${MO_FILE} DESTINATION ${CMAKE_INSTALL_LOCALEDIR})
INSTALL(TARGETS ../${PROJ} DESTINATION "bin")

#if(DEFINED DEBUG)
	find_package(Gettext REQUIRED)
	find_program(GETTEXT_XGETTEXT_EXECUTABLE xgettext)
	if(NOT GETTEXT_XGETTEXT_EXECUTABLE  OR NOT GETTEXT_MSGFMT_EXECUTABLE)
		  message(FATAL_ERROR "xgettext not found")
	endif()
add_custom_command(
	OUTPUT ${PO_FILE}
	COMMAND ${GETTEXT_XGETTEXT_EXECUTABLE} -D ${CMAKE_CURRENT_SOURCE_DIR} --from-code=koi8-r ${SOURCES} -c -k_ -kN_ -o ${PO_FILE}
	COMMAND sed 's/charset=UTF-8/charset=koi8-r/' ${PO_FILE} | enconv > tmp && mv -f tmp ${PO_FILE}
	COMMAND ${GETTEXT_MSGMERGE_EXECUTABLE} -Uis ${RU_FILE} ${PO_FILE}
	DEPENDS ${SOURCES})
add_custom_command(
	OUTPUT ${MO_FILE}
	COMMAND ${GETTEXT_MSGFMT_EXECUTABLE} ${RU_FILE} -o ${MO_FILE}
	DEPENDS ${RU_FILE})
#endif(DEFINED DEBUG)
