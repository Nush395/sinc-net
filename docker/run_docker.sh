#!/bin/bash

if [ -z "$1" ]
  then
    command="jupyter lab --ip 0.0.0.0 --no-browser"
else
	command="$1"
fi

docker run --gpus '"device=1"' -u $(id -u):$(id -g) \
		   -w /home/$(whoami)/ \
	       -v /home/$(whoami):/home/$(whoami) \
	       -v /scratch:/scratch \
		   -v /etc/passwd:/etc/passwd:ro \
		   -v /etc/group:/etc/group:ro \
		   -e TF_FORCE_GPU_ALLOW_GROWTH=true \
		   --rm -it \
		   -p 8888:8888 \
			sincnet:2.3 $command


# -v /home/$(whoami)/.keras:/tmp/.keras
