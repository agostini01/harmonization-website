Log into your VPN service.

Secure link steps
* Enter email username
* Enter password
* Get auth token from email
* Click connect
* Click Connect using the SecureLink Connection Manager
  * This will donwload an app
  * Open the app
  * Connection will be stabilished (windows requires confirmation)
* Options will appear, take note of the IP address of your target machine
  * To connect with ssh use `ssh <username>@<target-ip>`

Take note of the ip address of your target machine.

ssh into the machine

```bash
ssh <username>@<target-ip>
```

Create a development folder and download the project code

```
mkdir $HOME/Development
cd ~/Development
git clone https://github.com/agostini01/harmonization-website.git server

# Enter the development folder
cd ~/Development/server
```

Open a tmux session so that the terminal never dies in this machine.
See tmuxcheatsheet
```
# To open a session
tmux new -s server-sess

# To attach to a previously open session
tmux attach

# To attach to a specific session
tmux attach -t server-sess

# To detacth but not kill the session (for when you are done with changes)
```
Use the keyboard sequence: <kbd>ctrl</kbd>+<kbd>b</kbd> <kbd>d</kbd>

Make a copy of the sample config files and edit your configurations
```
cp config/.env.dev-sample config/.env.dev
cp config/.env-api.dev-sample config/.env-api.dev

# Open the file for edits with nano
nano config/.env.dev
# and
nano config/.env-api.dev

# Save and close the file with nano
```
Use the keyboard sequence: <kbd>ctrl</kbd>+<kbd>x</kbd> <kbd>y</kbd> <kbd>Enter</kbd>

Start the server
```
# To run on the background, use -d option
CURRENT_UID=$(id -u):$(id -g) DEV_UID=$(id -u) docker-compose up -d --build

# To run on the foreground, remove -d option
CURRENT_UID=$(id -u):$(id -g) DEV_UID=$(id -u) docker-compose up --build
```

At this point, the server will be running and available at port 80.
Return to secure link website and click on the `Access to a Web Server` link.
A Window saying welcome will be displayed.

Now lets create admin users. Back in the terminal. Divide the pane of the 
tmux session (in case -d option was not used)

Use the combination: <kbd>ctrl</kbd>+<kbd>b</kbd> <kbd>"</kbd>
```
# make sure you are in the project folder
cd ~/Development/server

# Now it is possible to swtich between panes with
```
Use the combination<kbd>ctrl</kbd>+<kbd>b</kbd> <kbd>o</kbd>


Run the commands and follow the promts.

```bash
# These two are necessary because 2 servers are running:
# web container runs the website frontend
docker-compose  exec web python manage.py createsuperuser

# api container runs the database and graph generation backend
docker-compose  exec api python manage.py createsuperuser
```