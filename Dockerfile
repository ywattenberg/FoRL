FROM cpacker/rl-generalization
RUN apt-get --allow-unauthenticated update && apt install ssh  -y
CMD ["service", "ssh", "start", "-D"]
