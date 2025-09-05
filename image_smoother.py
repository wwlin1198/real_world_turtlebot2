import numpy as np
import cv2 

class ImageSmoother():
    def __init__(self,flat_size = 7056, img_size = (84,84), threshold = 0.01):
        super(ImageSmoother, self).__init__()
        self.most_recent_nonzero = np.ones(flat_size)*0.8832035064697266
        self.flat_size = flat_size
        self.img_size = img_size
        self.t = threshold
        # print("IMAGE SMOOTHER ACTIVATE")

    def UpdateNonzero(self, obs):
        self.most_recent_nonzero[obs>self.t] = obs[obs>self.t]

    def step(self, action):
        obs, reward, term, trunc, info = self.env.step(action)
        obs = self.Smooth(obs)
        return obs, reward, term, trunc, info
    
    def reset(self, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        obs = self.Smooth(obs)
        return obs, info
    
    # def edge_detector(self, image):
    #     image = image*255
    #     img_blur = cv2.GaussianBlur(image,(3,3), 0,0) 
    #     img_blur = np.uint8(img_blur)
    #     edges = cv2.Canny(image=img_blur, threshold1=35, threshold2=40)
    #     edges = np.float32(edges)
    #     return edges / 255

    def edge_detector(self, image, use_depth=False):
        if use_depth:
            temp = np.copy(image)
        image = image*255
        img_blur = cv2.GaussianBlur(image,(3,3), 0,0)
        img_blur = np.uint8(img_blur)
        edges = cv2.Canny(image=img_blur, threshold1=35, threshold2=45)
        edges = np.float32(edges)/255
        if use_depth:
            edges *= temp
        return edges

    def close_gaps(self, image):
        image = np.uint8(image*255)
        
        # Larger kernel for more aggressive smoothing
        kernel = np.ones((8,8), np.uint8)
        
        # Close then open
        closing = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
        opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)
        
        # Optional: Apply median filter for additional noise reduction
        opening = cv2.medianBlur(opening, 5)
        
        opening = np.float32(opening)
        return opening / 255

    def Smooth(self, stack, use_depth=False):
        new_stack = np.zeros(len(stack))
        for i in range(len(stack)//self.flat_size):
            frame = np.copy(stack[self.flat_size*i : self.flat_size*(i+1)])
            
            # Ensure frame is a 1D array
            if isinstance(frame, np.ndarray) and frame.ndim > 1:
                frame = frame.flatten()
                
            # Reshape to image dimensions for the edge detection
            frame = frame.reshape(self.img_size)
            
            frame = self.close_gaps(frame)
            frame = self.edge_detector(frame, use_depth)
            # frame = self.edge_detector(frame)
            
            # Remove top 1/4 of the image AFTER edge detection to avoid boundary lines
            top_portion = int(self.img_size[0] * 1/4)
            
            frame[:top_portion, :] = 0
            
            # Flatten back for assignment
            frame = frame.flatten()
            
            new_stack[self.flat_size*i : self.flat_size*(i+1)] = frame
        return new_stack
    

# class ImageSmoother:
#     def __init__(self, flat_size = 7056, img_size = (84,84), threshold = 0.01):
#         self.most_recent_nonzero = np.ones(flat_size)*0.8832035064697266
#         self.flat_size = flat_size
#         self.img_size = img_size
#         self.t = threshold
#         # print("IMAGE SMOOTHER ACTIVATE")

#     def UpdateNonzero(self, obs):
#         self.most_recent_nonzero[obs>self.t] = obs[obs>self.t]

#     def step(self, action):
#         obs, reward, term, trunc, info = self.env.step(action)
#         obs = self.Smooth(obs)
#         return obs, reward, term, trunc, info

#     def reset(self, seed=None, options=None):
#         obs, info = self.env.reset(seed=seed, options=options)
#         obs = self.Smooth(obs)
#         return obs, info

#     def Smooth(self, stack, use_depth=False):
#         new_stack = np.zeros(len(stack))
#         for i in range(len(stack)//self.flat_size):
#             frame = np.copy(stack[self.flat_size*i : self.flat_size*(i+1)])
#             self.UpdateNonzero(frame)
#             frame[frame<=self.t] = self.most_recent_nonzero[frame<=self.t]
#             frame = cv2.medianBlur(np.reshape(frame, (84, 84)),1)
#             # Remove top 1/4 of the image AFTER edge detection to avoid boundary lines
#             top_portion = int(self.img_size[0] * 1/5)
            
#             frame[:top_portion, :] = 0
            
#             # Flatten back for assignment
#             frame = frame.flatten()
#             new_stack[self.flat_size*i : self.flat_size*(i+1)] = frame
#         return new_stack
