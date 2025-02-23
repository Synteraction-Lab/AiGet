import subprocess


def disable_sleep():
    # Enable MacBook to continue running the program when closing the lip
    subprocess.call('echo "YOUR_TERMINAL_PASSWORD" | sudo -S pmset -a disablesleep 1', shell=True)


def enable_sleep():
    # Resume Sleep
    subprocess.call('echo "YOUR_TERMINAL_PASSWORD" | sudo -S pmset -a disablesleep 0', shell=True)


def launch_pupil_capture():
    subprocess.call('echo "YOUR_TERMINAL_PASSWORD" | sudo -S "/Applications/Pupil Capture.app/Contents/MacOS/pupil_capture"',
                    shell=True)


def start_exp():
    disable_sleep()
    launch_pupil_capture()


def end_exp():
    enable_sleep()


if __name__ == '__main__':
    start_exp()
