import pygame # to make imagines 
import neat
import time 
import os # to get files 
import random 
#for the font
pygame.font.init()
# settings the dimensions for the screen 
win_width = 500
win_height= 800

gen = 0
# found the images with os, loaded the image, and scaling it by two 
bird_imgs = [pygame.transform.scale2x(pygame.image.load(os.path.join("imgs", "bird1.png"))), 
             pygame.transform.scale2x(pygame.image.load(os.path.join("imgs", "bird2.png"))),
             pygame.transform.scale2x(pygame.image.load(os.path.join("imgs", "bird3.png")))]

pipe_img = pygame.transform.scale2x(pygame.image.load(os.path.join("imgs", "pipe.png")))
base_img = pygame.transform.scale2x(pygame.image.load(os.path.join("imgs", "base.png")))
bg_img = pygame.transform.scale2x(pygame.image.load(os.path.join("imgs", "bg.png")))

#setting the font 
statFont = pygame.font.SysFont("comicsans", 50)

class Bird:
    imgs = bird_imgs 
    maxRotation = 25 # how much the bird is going to tilt 
    RotVel = 20 # how much were going to rotate 
    animationTime = 5 # how long were going to see the bird move 
    
    def __init__(self, x,y):
        self.x = x
        self.y = y
        self.tilt = 0
        self.tick_count = 0 # time (tick rate)
        self.vel = 0
        self.height = self.y
        self.img_count = 0
        self.img = self.imgs[0]
    
    def jump(self):
        #tilt and change height on jump 
        self.vel = -10.5
        self.tick_count = 0
        self.height = self.y
    
    def move(self):
        self.tick_count += 1
        # this changes the tilt of the bird every tick
        d = self.vel*self.tick_count + 1.5 * self.tick_count**2
        # d being the displacement of the bird from tilting 
        # changes d if it is too much
        if d >= 16:
            d = 16
        if d < 0:
            d -= 2
        self.y += d
        if d< 0 or self.y < self.height + 50:
            if self.tilt < self.maxRotation:
                self.tilt = self.maxRotation
            else:
                if self.tilt > - 90:
                    self.tilt -= self.RotVel
        
    def draw(self, win): 
        self.img_count += 1
        # every update it flaps wings 
        if self.img_count < self.animationTime:
            self.img = self.imgs[0]
        elif self.img_count < self.animationTime*2:
            self.img = self.imgs[1]
        elif self.img_count < self.animationTime*3:
            self.img = self.imgs[2]
        elif self.img_count < self.animationTime*4:
            self.img = self.imgs[1]
        elif self.img_count < self.animationTime*4 + 1:
            self.img = self.imgs[0]
            self.img_count = 0
        
        if self.tilt <= -80:
            self.img = self.imgs[1]
            self.img_count = self.animationTime

        #rotates the bird at the center
        rotated_image = pygame.transform.rotate(self.img, self.tilt)
        new_rect = rotated_image.get_rect(center = self.img.get_rect(topleft = (self.x, self.y)).center)
        win.blit(rotated_image, new_rect.topleft)

    def get_mask(self):
        return pygame.mask.from_surface(self.img)

class Pipe:
    vel  = 5
    
    def __init__(self,x):
        self.x = x
        self.height = 0
        self.gap = 200
        
        self.top = 0
        self.bottom = 0
        self.pipeTop = pygame.transform.flip(pipe_img, False, True) # upside down img
        self.pipeBottom  = pipe_img
        
        self.passed = False
        self.set_height()
    
    def set_height(self):
        self.height = random.randrange(50, 450)
        #top left is the origin 
        self.top = self.height - self.pipeTop.get_height()
        self.bottom = self.height + self.gap
    
    def move(self):
        self.x -= self.vel
    
    def draw(self,win):
        win.blit(self.pipeTop, (self.x, self.top))
        win.blit(self.pipeBottom, (self.x, self.bottom))
    
    def collide(self, bird):
        #mask is a 2d list of the pixels from the hit box for pixel perfect
        birdMask = bird.get_mask()
        topMask = pygame.mask.from_surface(self.pipeTop)
        bottomMask = pygame.mask.from_surface(self.pipeBottom)
        
        topOffset = (self.x - bird.x, self.top - round(bird.y))
        bottomOffset = (self.x - bird.x, self.bottom - round(bird.y))
        
        bpoint = birdMask.overlap(bottomMask, bottomOffset)
        tpoint = birdMask.overlap(topMask, topOffset)

        if tpoint or bpoint:
            return True

        return False
        
class Base:
    vel = 5
    width = base_img.get_width()
    img = base_img
    
    def __init__(self, y):
        self.y = y
        self.x1 = 0
        self.x2 = self.width
        
        
    def move(self):
        self.x1 -= self.vel
        self.x2 -= self.vel
        
        if self.x1 + self.width < 0:
            self.x1 = self.x2 + self.width
        
        if self.x2 + self.width < 0:
            self.x2 = self.x1 + self.width 
    
    def draw(self, win):
        win.blit(self.img, (self.x1, self.y))
        win.blit(self.img, (self.x2, self.y))
        
def draw_window(win, birds, pipes, Base, score, gen):
    #draws the bg and updates the bird 
    win.blit(bg_img, (0,0))
    #blit means draw
    for pipe in pipes:
        pipe.draw(win)
    
    text = statFont.render("Score: " + str(score), 1, (255,255,255))
    win.blit(text, (win_width - 10 - text.get_width(), 10))
    
    text = statFont.render("Gen: " + str(gen), 1, (255,255,255))
    win.blit(text, (10, 10))
    
    Base.draw(win)
    
    for bird in birds:
        bird.draw(win)
    pygame.display.update()
    
def main(genomes, config):
    global gen
    gen += 1
    nets = []
    ge = []
    birds = []
    
    for _, g in genomes:
        net = neat.nn.FeedForwardNetwork.create(g, config)
        nets.append(net)
        birds.append(Bird(230, 350))
        g.fitness = 0
        ge.append(g)
        
    base = Base(730)
    pipes = [Pipe(600)]
    win = pygame.display.set_mode((win_width, win_height))
    clock = pygame.time.Clock()
    
    score = 0
    run = True
    while run:
        clock.tick(30)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                pygame.quit()
                quit()
                
        pipeInd = 0
        if len(birds) > 0:
            if len(pipes) > 1 and birds[0].x > pipes[0].x + pipes[0].pipeTop.get_width():
                pipeInd = 1   
        
        else:
            run = False
            break
        
        for x, bird in enumerate(birds):
            bird.move()
            ge[x].fitness += 0.1
            
            output = nets[x].activate((bird.y, abs(bird.y - pipes[pipeInd].height), abs(bird.y - pipes[pipeInd].bottom)))     
            
            if output[0] > 0.5:
                bird.jump()
        
        addPipe = False
        rem = []
        for pipe in pipes:
            for x, bird in enumerate(birds):
                if pipe.collide(bird):
                    ge[x].fitness -= 1
                    birds.pop(x)
                    nets.pop(x)
                    ge.pop(x)
                    
                if not pipe.passed and pipe.x < bird.x:
                    pipe.passed = True
                    addPipe = True
                    
            if pipe.x + pipe.pipeTop.get_width() < 0:
                rem.append(pipe)
                
            pipe.move()
        
        if addPipe:
            score += 1
            for g in ge:
                g.fitness += 5
            pipes.append(Pipe(600))
        
        for r in rem:
            pipes.remove(r)
            
        for x, bird in enumerate(birds):
            if bird.y + bird.img.get_height() >= 730 or bird.y < 0:
                birds.pop(x)
                nets.pop(x)
                ge.pop(x)
        
        base.move()
        draw_window(win,birds, pipes, base, score, gen)




def run(config_path):
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)

    p = neat.Population(config)
    
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    
    winner = p.run(main,50)
    
if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)           
    config_path = os.path.join(local_dir, "neat.txt") 
    run(config_path)

