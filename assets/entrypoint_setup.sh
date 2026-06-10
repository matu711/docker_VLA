#! /bin/bash

check_envs () {
    DOCKER_CUSTOM_USER_OK=true;
    if [ -z ${DOCKER_USER_NAME+x} ]; then 
        DOCKER_CUSTOM_USER_OK=false;
        return;
    fi
    
    if [ -z ${DOCKER_USER_ID+x} ]; then 
        DOCKER_CUSTOM_USER_OK=false;
        return;
    else
        if ! [ -z "${DOCKER_USER_ID##[0-9]*}" ]; then 
            echo -e "\033[1;33mWarning: User-ID should be a number. Falling back to defaults.\033[0m"
            DOCKER_CUSTOM_USER_OK=false;
            return;
        fi
    fi
    
    if [ -z ${DOCKER_USER_GROUP_NAME+x} ]; then 
        DOCKER_CUSTOM_USER_OK=false;
        return;
    fi

    if [ -z ${DOCKER_USER_GROUP_ID+x} ]; then 
        DOCKER_CUSTOM_USER_OK=false;
        return;
    else
        if ! [ -z "${DOCKER_USER_GROUP_ID##[0-9]*}" ]; then 
            echo -e "\033[1;33mWarning: Group-ID should be a number. Falling back to defaults.\033[0m"
            DOCKER_CUSTOM_USER_OK=false;
            return;
        fi
    fi
}


setup_env_user () {
    USER=$1
    USER_ID=$2
    GROUP=$3
    GROUP_ID=$4

    ## Create user
    useradd -m $USER

    ## Copy zsh/sh configs
    cp /root/.profile /home/$USER/
    cp /root/.bashrc /home/$USER/
    #cp /root/.zshrc /home/$USER/
    ## Copy terminator configs
    
	mkdir -p /home/$USER/.config/terminator
    cp /config /home/$USER/.config/terminator/config
    
	mkdir -p /root/.config/terminator
    cp /config /root/.config/terminator/config
    
	#cp -rf /root/.oh-my-zsh /home/$USER/
    #rm -rf /home/$USER/.oh-my-zsh/custom/pure.zsh-theme /home/$USER/.oh-my-zsh/custom/async.zsh
    #ln -s /home/$USER/.oh-my-zsh/custom/pure/pure.zsh-theme /home/$USER/.oh-my-zsh/custom/
    #ln -s /home/$USER/.oh-my-zsh/custom/pure/async.zsh /home/$USER/.oh-my-zsh/custom/
    #sed -i -e 's@ZSH=\"/root@ZSH=\"/home/$USER@g' /home/$USER/.zshrc
    # Copy SSH keys & fix owner
    if [ -d "/root/.ssh" ]; then
        cp -rf /root/.ssh /home/$USER/
        chown -R $USER:$GROUP /home/$USER/.ssh
    fi
    
    ## Fix owner
    chown $USER:$GROUP /home/$USER
    chown -R $USER:$GROUP /home/$USER/.config
    chown $USER:$GROUP /home/$USER/.profile
    chown $USER:$GROUP /home/$USER/.bashrc
    #chown $USER:$GROUP /home/$USER/.zshrc
    #chown -R $USER:$GROUP /home/$USER/.oh-my-zsh

    ## This a trick to keep the evnironmental variables of root which is important!
    echo "if ! [ \"$DOCKER_USER_NAME\" = \"$(id -un)\" ]; then" >> /root/.bashrc
    echo "    cd /home/$DOCKER_USER_NAME" >> /root/.bashrc
    echo "    su $DOCKER_USER_NAME" >> /root/.bashrc
    echo "fi" >> /root/.bashrc

    #echo "if ! [ \"$DOCKER_USER_NAME\" = \"$(id -un)\" ]; then" >> /root/.zshrc
    #echo "    cd /home/$DOCKER_USER_NAME" >> /root/.zshrc
    #echo "    su $DOCKER_USER_NAME" >> /root/.zshrc
    #echo "fi" >> /root/.zshrc

    ## Setup Password-file
    PASSWDCONTENTS=$(grep -v "^${USER}:" /etc/passwd)
    GROUPCONTENTS=$(grep -v -e "^${GROUP}:" -e "^docker:" /etc/group)

    (echo "${PASSWDCONTENTS}" && echo "${USER}:x:$USER_ID:$GROUP_ID::/home/$USER:/bin/bash") > /etc/passwd
    (echo "${GROUPCONTENTS}" && echo "${GROUP}:x:${GROUP_ID}:") > /etc/group
    (if test -f /etc/sudoers ; then echo "${USER}  ALL=(ALL)   NOPASSWD: ALL" >> /etc/sudoers ; fi)


  ### Tahara
	# Authority issues
  sudo adduser $USER video
  sudo chmod 777 /root

  chown $USER /home/$USER/.pyenv/shims
  chown $USER /home/$USER/.pyenv/version
  chown $USER /home/$USER/.ipython -R
  # chmod 755 /home/$USER/.ipython/ -r
  # chown hikaru-s /home/hikaru-s/.pyenv/shims
          # ROS
    echo "source /home/$USER/catkin_ws/devel/setup.bash" >> /root/.bashrc
    sudo mkdir /home/$USER/.ros/
    sudo chmod -R 777 /home/$USER/.ros/
    echo "source /home/$USER/catkin_ws/devel/setup.bash" >> /root/.bashrc
    sudo mkdir /home/$USER/.rviz/
    sudo chmod -R 777 /home/$USER/.rviz/
    # MuJoCo & mujoco-py 
    export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/home/$USER/.mujoco/mujoco200/bin
    sudo cp -r /root/.mujoco/ /home/$USER/
    sudo chmod -R 777 /home/$USER/.mujoco/
    sudo chmod -R 777 /usr/local/lib/python3.8/dist-packages/mujoco_py*
    sudo mkdir /home/$USER/.cache
    sudo chmod -R 777 /home/$USER/.cache/

    # pygame
    sudo chmod -R 777 /home/$USER/.config/
    sudo chmod -R 777 /dev/input

    # Isaac
    sudo cp -r /root/isaacgym/ /home/$USER/isaacgym
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/shunsuke-a/.mujoco/mujoco210/bin
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia

    sudo mkdir /home/$USER/.local/
    sudo chmod -R 777 /home/$USER/.local/


    sudo touch /home/$USER/.pdbhistory
    sudo chmod -R 777 /home/$USER/.pdbhistory
}




# ---Main---

# Create new user
## Check Inputs
check_envs

## Determine user & Setup Environment
if [ $DOCKER_CUSTOM_USER_OK == true ]; then
    echo "  -->DOCKER_USER Input is set to '$DOCKER_USER_NAME:$DOCKER_USER_ID:$DOCKER_USER_GROUP_NAME:$DOCKER_USER_GROUP_ID'";
    echo -e "\033[0;32mSetting up environment for user=$DOCKER_USER_NAME\033[0m"
    setup_env_user $DOCKER_USER_NAME $DOCKER_USER_ID $DOCKER_USER_GROUP_NAME $DOCKER_USER_GROUP_ID
else
    echo "  -->DOCKER_USER* variables not set. Using 'root'.";
    echo -e "\033[0;32mSetting up environment for user=root\033[0m"
    DOCKER_USER_NAME="root"
fi


# Change shell to zsh
### chsh -s /usr/bin/zsh $DOCKER_USER_NAME

# Run CMD from Docker
"$@"
