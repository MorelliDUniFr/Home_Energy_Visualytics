# Home Energy Visualytics

# Table of Contents
TODO

## 1. Introduction
TODO

## 2. Hardware Setup
### 2.1 Configuration of Raspberry Pi via the Pi imager
If you don't have the Raspberry Pi Imager installed, you can download it from the official Raspberry Pi website. The imager allows you to easily set up your Raspberry Pi with the desired operating system and configurations.
1. Download the Raspberry Pi Imager from the official website: https://www.raspberrypi.com/software/
1. Select Device: Raspberry Pi 5
1. OS: Raspberry Pi OS Lite (64-bit)
1. Storage: 64 GB (Should be the SD card)

Click next and then select the following options (under _modify settings_):
1. Under general: 
	1. Hostname: raspberrypi
    1. Username: morel
    1. Password: morel
    1. Wifi: Enable wifi and enter the credentials
    1. Nation: CH
    1. Time Zone: Europe/Zurich
	1. Layout keyboard: ch
1. Under services:
   1. SSH: Enable SSH

Once is all done, click on _Save_ and then _Write_ to flash the SD card with the selected configurations. This process may take a few minutes.

### 2.2 Setting up the gPlug
This set up is very simple, just plug the gPlug into the wall socket and connect it to the desired Wi-Fi network (must be the same as the Raspberry Pi).

## 3. Software Setup
### 3.1 Access to Raspberry Pi
```bash
ssh morel@raspberrypi.local
```
And then insert the password set before.

### 3.2 Install dependencies
```bash
sudo apt update
```
```bash
sudo apt upgrade
```

it can take a while...

### 3.3 Installing docker
```bash
sudo apt-get update
```
```bash
sudo apt-get install ca-certificates curl
```
```bash
sudo install -m 0755 -d /etc/apt/keyrings
```
```bash
sudo curl -fsSL https://download.docker.com/linux/debian/gpg -o /etc/apt/keyrings/docker.asc
```
```bash
sudo chmod a+r /etc/apt/keyrings/docker.asc
```
Add the repository to Apt sources:
```bash
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/debian \
  $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
```
```bash
sudo apt-get update
```
Install the latest version:
```bash
sudo apt-get install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
```
Verify the installation:
```bash
sudo docker run hello-world
```

### 3.4 Prepare the environment
Create a directory for the project:
```bash
sudo mkdir /home/pi
```

Give full access (to be able to access the files from the host):
```bash
sudo chown morel:morel /home/pi
```
```bash
sudo usermod -aG docker morel
```
```bash
sudo reboot
```

### 3.5 Installing Docker compose
```bash
sudo apt-get update
```
```bash
sudo apt-get install docker-compose-plugin
```
```bash
docker compose version
```
