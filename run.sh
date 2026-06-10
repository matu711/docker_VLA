export USR_NAME=shunsuke-m

echo $DISPLAY


xhost local:root
xhost +local:docker


docker run --rm -it \
	--name vla-env-image2 \
	-w /home/${USR_NAME}/docker_VLA \
	--volume /home/${USR_NAME}/docker_VLA:/home/${USR_NAME}/docker_VLA \
	--volume $HOME/.Xauthority:/home/$(id -un)/.Xauthority -e XAUTHORITY=/home/$(id -un)/.Xauthority \
	--volume /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=unix$DISPLAY \
	--privileged --net=host --ipc=host \
	--gpus all \
	-e DISPLAY=$DISPLAY \
	-v /tmp/.X11-unix:/tmp/.X11-unix \
	-e DOCKER_USER_NAME=$(id -un) \
	-e DOCKER_USER_ID=$(id -u) \
	-e DOCKER_USER_GROUP_NAME=$(id -gn) \
	-e DOCKER_USER_GROUP_ID=$(id -g) \
	-v /home/${USR_NAME}/docker_VLA/workspace:/home/${USR_NAME}/docker_VLA/workspace \
	docker-vla-env-template
	# --gpus all \
	# --device /dev/input/js1:/dev/input/js1 \
	# --device /dev/input/js2:/dev/input/js2 \
	# --device /dev/ttyUSB0:/dev/ttyUSB0 \
	# --device /dev/ttyACM0:/dev/ttyACM0 \