CPP = hipcc
LFLAGS = -lX11 -lpthread
CFLAGS = -std=c++11

SOURCES = $(wildcard *.cpp)
OBJECTS = $(subst .cpp,.o,$(SOURCES))

edge-detection.ex: $(OBJECTS)
	$(CPP) $(CFLAGS) $(LFLAGS) -o $@ $(OBJECTS) 

%.o: %.cpp
	$(CPP) $(CFLAGS) -c $<

.PHONY: clean

clean:
	rm -f edge-detection.ex $(OBJECTS) *.out *.err
	rm -f ../../images/output/*.bmp
