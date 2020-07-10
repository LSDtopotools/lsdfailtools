# make with make -f test_iverson.make

CC=g++
CFLAGS=-c -Wall -O3 -g -fopenmp -std=c++11
OFLAGS = -Wall -O3 -g -fopenmp -std=c++11
LDFLAGS= -Wall
SOURCES=Iverson_Model.cpp \
    ../LSDPorewaterColumn.cpp \
    ../LSDPorewaterParams.cpp \
    ../LSDParameterParser.cpp \
    ../LSDMostLikelyPartitionsFinder.cpp \
    ../LSDRasterInfo.cpp \
    ../LSDIndexRaster.cpp \
    ../LSDRaster.cpp \
    ../LSDShapeTools.cpp \
    ../LSDStatsTools.cpp

OBJECTS=$(SOURCES:.cpp=.o)
EXECUTABLE=Iverson_Model.exe

all: $(SOURCES) $(EXECUTABLE)

$(EXECUTABLE): $(OBJECTS)
	$(CC) $(OFLAGS) $(OBJECTS) -o $@

.cpp.o:
	$(CC) $(CFLAGS) $< -o $@
