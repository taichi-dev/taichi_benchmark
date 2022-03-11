CMAKE = cmake
RELEASE = -DCMAKE_BUILD_TYPE=Release
DEBUG = -DCMAKE_BUILD_TYPE=Debug
SRC = .
BUILD = cmake-build
TARGET = MPM3D

setup-release:
	${CMAKE} ${RELEASE} -S ${SRC} -B ${BUILD}-release 
	
run-release: setup-release
	${CMAKE} --build ${BUILD}-release  --target ${TARGET} 
	./${BUILD}-release/${TARGET} 
	
setup-debug:
	${CMAKE} ${DEBUG} -S ${SRC} -B ${BUILD}-debug 
	
run-debug: setup-debug
	${CMAKE} --build ${BUILD}-debug --target ${TARGET} 
	./${BUILD}-debug/${TARGET} 

run: run-release

clean: 
	rm -r -f ${BUILD}-debug ${BUILD}-release
