
import torch
if(torch.cuda.is_available()):
  import torch.cuda as t
else:
  import torch
import torch.nn as nn
import torch.nn.functional as F


class BCL(nn.Module):
    def __init__(self, N, C, Cp):
        super(BCL, self).__init__()
        print("Init")
        #N = size of grid
        self.N = N
        #Size of outgoing channel
        self.C = C
        #Size of incoming channel
        self.Cp = Cp         
        self.conv3d = nn.Conv3d(in_channels=self.Cp , out_channels=self.C, kernel_size=(3,3,3), stride=1, padding=0, bias=True)
        self.num_points = 1500
        
    def forward(self, data, prev_features):
        
        #print("Inside BCL Forward")
        trilinear, dist, xc, yc, zc = self.splat(data, prev_features)
        #print("Trillinear output: ", trilinear.shape)
        
        conv_out = self.conv3d(trilinear)        
        #print("BCL conv out: ", conv_out.shape)

        slice_out = self.slice(conv_out, dist, xc, yc, zc)
        #print("Slice out: ", slice_out.shape)
        return slice_out
        
    
    def splat(self, data, prev_features):
        epsilon = 0.0001
        
        #print("Inside Splat")
        #print('Original data: ', data.shape)
        #print('Conv prev layer output: ', prev_features.shape)
        num_channels = prev_features.shape[1]
        #print("Num channel", num_channels)
        
        N = self.N
        trilinear = torch.zeros((1, num_channels, N,N,N))
        trilinear_count = torch.zeros((N, N, N)) + epsilon
        grid_size = (N,N,N)

        max_dimensions = torch.ceil(torch.max(data, 0)[0])
        #print("Max:", max_dimensions)
        min_dimensions = torch.floor(torch.min(data, 0)[0])
        #print("Min:", min_dimensions)
        bounding_box_dimensions = max_dimensions - min_dimensions 
        grid_spacing = (bounding_box_dimensions)/(N-9)
        max_grid_dist = 1.73 * grid_spacing
        #print('Grid spacing: ', grid_spacing)
        
        x = torch.linspace(min_dimensions[0]-grid_spacing[0]*4, max_dimensions[0]+grid_spacing[0]*4, N)
        y = torch.linspace(min_dimensions[1]-grid_spacing[1]*4, max_dimensions[1]+grid_spacing[1]*4, N)
        z = torch.linspace(min_dimensions[2]-grid_spacing[2]*4, max_dimensions[2]+grid_spacing[2]*4, N)

        xc = torch.tensor(torch.floor((data[:,0] - x[0])/grid_spacing[0]), dtype=torch.long)
        yc = torch.tensor(torch.floor((data[:,1] - y[0])/grid_spacing[1]), dtype=torch.long)
        zc = torch.tensor(torch.floor((data[:,2] - z[0])/grid_spacing[2]), dtype=torch.long)
        assert(xc.shape[0] == data.shape[0]) #Number of points per example
        #print(xc)

        xt = torch.tensor(torch.cat((x[xc].reshape(self.num_points, 1), x[xc+1].reshape(self.num_points, 1)), dim=1), dtype=torch.long)
        yt = torch.tensor(torch.cat((y[yc].reshape(self.num_points, 1), y[yc+1].reshape(self.num_points, 1)), dim=1), dtype=torch.long)
        zt = torch.tensor(torch.cat((z[zc].reshape(self.num_points, 1), z[zc+1].reshape(self.num_points, 1)), dim=1), dtype=torch.long)

        dist = torch.zeros((self.num_points, 8))
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    a000 = torch.cat([(xt[:,i]).reshape(self.num_points,1),
                                (yt[:,j]).reshape(self.num_points,1), 
                                (zt[:,k]).reshape(self.num_points,1)], dim=1)
                    a000 = torch.tensor(a000, dtype=torch.float)
                    dist[:,4*i + 2*j + k] = torch.norm(data - a000, dim=1)
                    trilinear_count[xc+k, yc+j, zc+i] +=1

        #print("tri: ", trilinear.shape)
        temp = torch.zeros((num_channels, N, N, N))
        
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    temp[:,xc+i, yc+j, zc+k] += prev_features.reshape(num_channels, self.num_points) /dist[:,4*i + 2*j + k]
                    trilinear = temp.reshape(1,num_channels, N, N, N)
       
        #Add trilinear_count
        #print("trilinear", trilinear.shape)
        #print("trilinear count", trilinear_count.shape)
        final_splat = trilinear
        
        return final_splat, dist, xc, yc, zc
      
      
    def slice(self, trilinear, dist, xc, yc, zc):
        
        #print("Inside Slice")
        slice_out = torch.zeros((1,trilinear.shape[1], self.num_points, 1, 1))
        temp = torch.zeros((1,trilinear.shape[1],self.num_points))

        for i in range(2):
            for j in range(2):
                for k in range(2):
                    temp +=  trilinear[:,:,xc+i, yc+j, zc+k]*dist[:, 4*i + 2*j * k]
        slice_out = (temp/8).reshape(1,trilinear.shape[1], self.num_points, 1,1)
        
        return slice_out
      
      
    