CXX ?= g++
CXXFLAGS ?= -O3 -std=c++20 -Wall -Wextra -Wpedantic

.PHONY: all test sanitize ubsan clean wide test-wide ubsan-wide
all: v2 build/hyqec-check-reference build/hyqec-check-native-v1.3.0 build/hyqec-check-wide-native-v1.3.0

# Release builds never overwrite the historical pinned executables.
wide: build/hyqec-check-wide-native-v1.3.0

build/hyqec-check-wide-native-v1.3.0: native/checker_wide.cpp
	mkdir -p build
	$(CXX) $(CXXFLAGS) -fopenmp $< -o $@

test-wide: all wide
	python3 -m unittest discover -s tests -p 'test_wide.py' -v

ubsan-wide:
	mkdir -p build
	$(CXX) -O1 -g -std=c++20 -Wall -Wextra -D_GLIBCXX_ASSERTIONS -fsanitize=undefined -fsanitize-undefined-trap-on-error -fopenmp native/checker_wide.cpp -o build/hyqec-check-wide-native-v1.3.0-ubsan

build/hyqec-check-native-v1.3.0: native/checker.cpp
	mkdir -p build
	$(CXX) $(CXXFLAGS) -fopenmp $< -o $@

build/hyqec-check-reference: native/reference.cpp
	mkdir -p build
	$(CXX) $(CXXFLAGS) -fopenmp $< -o $@

test: all wide build/v2-node-driver
	python3 -m unittest discover -s tests -v

sanitize:
	mkdir -p build
	$(CXX) -O1 -g -std=c++20 -Wall -Wextra -fsanitize=address,undefined -fno-omit-frame-pointer -fopenmp native/reference.cpp -o build/hyqec-check-reference-sanitize

ubsan:
	mkdir -p build
	$(CXX) -O1 -g -std=c++20 -Wall -Wextra -D_GLIBCXX_ASSERTIONS -fsanitize=undefined -fsanitize-undefined-trap-on-error -fopenmp native/reference.cpp -o build/hyqec-check-reference-ubsan
	$(CXX) -O1 -g -std=c++20 -Wall -Wextra -D_GLIBCXX_ASSERTIONS -fsanitize=undefined -fsanitize-undefined-trap-on-error -fopenmp native/checker.cpp -o build/hyqec-check-native-v1.3.0-ubsan

clean:
	rm -rf build/hyqec-check-native-v1.3.0* build/hyqec-check-wide-native-v1.3.0* build/hyqec-check-reference*

.PHONY: v2 ubsan-v2 sanitize-v2
V2_HEADERS = native/v2_types.hpp native/v2_decide.hpp native/v2_verify.hpp
v2: build/hyqec-check-v2-native-v2.0.0

build/hyqec-check-v2-native-v2.0.0: native/checker_v2.cpp $(V2_HEADERS)
	mkdir -p build
	$(CXX) $(CXXFLAGS) -fopenmp $< -o $@

ubsan-v2:
	mkdir -p build
	$(CXX) -O1 -g -std=c++20 -Wall -Wextra -D_GLIBCXX_ASSERTIONS -fsanitize=undefined -fsanitize-undefined-trap-on-error -fopenmp native/checker_v2.cpp -o build/hyqec-check-v2-native-v2.0.0-ubsan

sanitize-v2:
	mkdir -p build
	$(CXX) -O1 -g -std=c++20 -Wall -Wextra -D_GLIBCXX_ASSERTIONS -fsanitize=address,undefined -fno-omit-frame-pointer -fopenmp native/checker_v2.cpp -o build/hyqec-check-v2-native-v2.0.0-sanitize

.PHONY: test-v2 v2-wrong-j
build/v2-node-driver: tests/v2_node_driver.cpp tests/v2_verify_regressions.hpp native/checker_v2.cpp $(V2_HEADERS)
	$(CXX) $(CXXFLAGS) -Wno-unused-function -fopenmp $< -o $@

test-v2: v2 build/v2-node-driver
	python3 -m unittest discover -s tests -p 'test_v2*.py' -v

v2-wrong-j:
	$(CXX) $(CXXFLAGS) -DHYQEC_V2_WRONG_J -fopenmp native/checker_v2.cpp -o build/hyqec-check-v2-native-wrong-j

# Clang supplies ASan where the system GCC installation lacks libasan.
.PHONY: sanitize-v2-clang
sanitize-v2-clang:
	clang++ -O1 -g -std=c++20 -Wall -Wextra -D_GLIBCXX_ASSERTIONS -fsanitize=address,undefined -fno-omit-frame-pointer -fno-pie -no-pie -fopenmp native/checker_v2.cpp -o build/hyqec-check-v2-native-v2.0.0-sanitize
	clang++ -O1 -g -std=c++20 -Wall -Wextra -Wno-unused-function -D_GLIBCXX_ASSERTIONS -DHYQEC_V2_ZERO_KAPPA -fsanitize=address,undefined -fno-omit-frame-pointer -fno-pie -no-pie -fopenmp tests/v2_node_driver.cpp -o build/v2-node-driver-collisions-sanitize
