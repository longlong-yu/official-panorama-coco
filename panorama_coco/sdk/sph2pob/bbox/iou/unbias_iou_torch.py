import torch


class Sph:
    '''Unbiased IoU Computation for Spherical Rectangles'''

    def __init__(self):
        self.visited, self.trace, self.pot = [], [], []

    def area(self, fov_x, fov_y):
        '''Area Computation'''
        return 4 * torch.arccos(-torch.sin(fov_x / 2) * torch.sin(fov_y / 2)) - 2 * torch.pi

    def getNormal(self, bbox):
        '''Normal Vectors Computation'''
        theta, phi, fov_x_half, fov_y_half = bbox[:, [
            0]], bbox[:, [1]], bbox[:, [2]] / 2, bbox[:, [3]] / 2
        V_lookat = torch.cat((
            torch.sin(phi) * torch.cos(theta), torch.sin(phi) *
            torch.sin(theta), torch.cos(phi)
        ), dim=1)
        V_right = torch.cat(
            (-torch.sin(theta), torch.cos(theta), torch.zeros(theta.shape, device=theta.device)), dim=1)
        V_up = torch.cat((
            -torch.cos(phi) * torch.cos(theta), -torch.cos(phi) *
            torch.sin(theta), torch.sin(phi)
        ), dim=1)
        N_left = -torch.cos(fov_x_half) * V_right + torch.sin(fov_x_half) * V_lookat
        N_right = torch.cos(fov_x_half) * V_right + torch.sin(fov_x_half) * V_lookat
        N_up = -torch.cos(fov_y_half) * V_up + torch.sin(fov_y_half) * V_lookat
        N_down = torch.cos(fov_y_half) * V_up + torch.sin(fov_y_half) * V_lookat
        V = torch.stack([
            torch.cross(N_left, N_up, dim=-1), torch.cross(N_down, N_left, dim=-1),
            torch.cross(N_up, N_right, dim=-1), torch.cross(N_right, N_down, dim=-1)
        ])
        norm = torch.linalg.norm(V, dim=2)[
            :, :, None].repeat_interleave(V.shape[2], dim=2)
        V = torch.true_divide(V, norm)
        E = torch.stack([
            torch.stack([N_left, N_up]), torch.stack([N_down, N_left]), 
            torch.stack([N_up, N_right]), torch.stack([N_right, N_down])
        ])
        return torch.stack([N_left, N_right, N_up, N_down]), V, E

    def interArea(self, orders, E):
        '''Intersection Area Computation'''
        angles = -torch.matmul(E[:, 0, :][:, None, :],
                            E[:, 1, :][:, :, None])
        angles = torch.clip(angles, -1, 1)
        whole_inter = torch.arccos(angles)
        inter_res = torch.zeros(orders.shape[0],device=E.device)
        loop = 0
        idx = torch.where(orders != 0)[0]
        iters = orders[idx]
        for i, j in enumerate(iters):
            inter_res[idx[i]] = torch.sum(
                whole_inter[loop:loop+j], dim=0) - (j - 2) * torch.pi
            loop += j
        return inter_res

    def dfs(self, node_index, node_graph):
        '''Find Cycles by DFS'''
        if node_index in self.trace:
            trace_index = self.trace.index(node_index)
            self.pot.append(self.trace[trace_index:len(self.trace)])
            return
        self.visited.append(node_index)
        self.trace.append(node_index)
        if(node_index != ''):
            children = node_graph[node_index].split('#')
            for child in children:
                self.dfs(child, node_graph)
            self.trace.pop()
            self.visited.pop()

    def remove_redundant_points_by_DFS(self, points, edges):
        #points = points.cpu().numpy()
        #edges = edges.cpu().numpy()
        '''Remove redundant Points'''
        serial, reverse_serial, nodes_list = {}, {}, {}
        number = 0
        for i in range(points.shape[0]):
            first, second = tuple(edges[i, 0, :].cpu().numpy()), tuple(edges[i, 1, :].cpu().numpy())
            if first not in serial.keys():
                serial[first] = number
                reverse_serial[str(number)] = edges[i, 0, :]
                number += 1
            if second not in serial.keys():
                serial[second] = number
                reverse_serial[str(number)] = edges[i, 1, :]
                number += 1
            if str(serial[first]) not in nodes_list.keys():
                nodes_list[str(serial[first])] = str(serial[second])
            elif str(serial[second]) not in nodes_list[str(serial[first])].split('#') and serial[second] != serial[first]:
                nodes_list[str(serial[first])] += '#' + str(serial[second])

        for i in range(points.shape[0]):
            self.dfs(str(i), nodes_list)
            self.trace.clear()
            self.visited.clear()
            if len(self.pot) != 0:
                break

        next_nodes_list = [self.pot[0][-1]] + self.pot[0][:-1]
        true_edges = torch.stack([torch.stack([reverse_serial[e], reverse_serial[p]])
                               for e, p in zip(self.pot[0], next_nodes_list)])
        true_inter = self.interArea(torch.tensor([len(self.pot[0])]), true_edges)
        self.pot.clear()
        return true_inter

    def remove_outer_points(self, dets, gt):
        '''Remove points outside the two spherical rectangles'''
        N_dets, V_dets, E_dets = self.getNormal(dets)
        N_gt, V_gt, E_gt = self.getNormal(gt)
        N_res = torch.vstack((N_dets, N_gt))
        V_res = torch.vstack((V_dets, V_gt))
        E_res = torch.vstack((E_dets, E_gt))

        N_dets_expand = N_dets.repeat_interleave(N_gt.shape[0], dim=0)
        N_gt_expand = N_gt.repeat((N_dets.shape[0], 1, 1))

        tmp1 = torch.cross(N_dets_expand, N_gt_expand, dim=-1)
        mul1 = torch.true_divide(
            tmp1, torch.linalg.norm(tmp1, dim=2)[:, :, None].repeat_interleave(tmp1.shape[2], dim=2) + 1e-10)

        tmp2 = torch.cross(N_gt_expand, N_dets_expand, dim=-1)
        mul2 = torch.true_divide(
            tmp2, torch.linalg.norm(tmp2, dim=2)[:, :, None].repeat_interleave(tmp2.shape[2], dim=2) + 1e-10)

        V_res = torch.vstack((V_res, mul1))
        V_res = torch.vstack((V_res, mul2))

        dimE = (E_res.shape[0] * 2, E_res.shape[1],
                E_res.shape[2], E_res.shape[3])
        E_res = torch.vstack(
            (E_res, torch.hstack((N_dets_expand, N_gt_expand)).reshape(dimE)))
        E_res = torch.vstack(
            (E_res, torch.hstack((N_gt_expand, N_dets_expand)).reshape(dimE)))

        res = torch.matmul(V_res.permute(
            (1, 0, 2)).contiguous(), N_res.permute((1, 2, 0)).contiguous())
        
        res = torch.round(res, decimals=8)
        value = torch.all(res >= 0, dim=2)
        return value, V_res, E_res

    def computeInter(self, dets, gt):
        '''
        The whole Intersection Area Computation Process (3 Steps):
        Step 1. Compute normal vectors and point vectors of each plane for eight boundaries of two spherical rectangles;
        Step 2. Remove unnecessary points by two Substeps:
           - Substep 1: Remove points outside the two spherical rectangles;
           - Substep 2: Remove redundant Points. (This step is not required for most cases that do not have redundant points.)
        Step 3. Compute all left angles and the final intersection area.
        '''
        value, V_res, E_res = self.remove_outer_points(dets, gt)
        ind0 = torch.where(value)[0]
        ind1 = torch.where(value)[1]

        if ind0.shape[0] == 0:
            return torch.zeros((dets.shape[0]), device=dets.device)

        E_final = E_res[ind1, :, ind0, :]
        orders = torch.bincount(ind0)

        minus = dets.shape[0] - orders.shape[0]
        if minus > 0:
            orders = torch.nn.functional.pad(orders, (0, minus), mode='constant')

        V_res = torch.round(V_res, decimals=8)
        split_vectors = torch.split(V_res.permute((1, 0, 2)).contiguous()[
                                 value, :], orders.tolist())
        flag = torch.zeros(len(split_vectors), device=dets.device)
        #print(len(split_vectors))
        # TODO It's a big loop with very low speed, and we haven't figured out how to eliminate it
        for i, vec in enumerate(split_vectors):
            unique_v = torch.unique(vec, return_counts=True)[1]
            if unique_v.sum() != len(unique_v):
                flag[i] = 1

        inter = self.interArea(orders, E_final)


        if False:#flag.sum():
            split_edges = torch.split(E_final, orders.tolist())
            for ind in torch.where(flag)[0]:
                inter_area_redundant = self.remove_redundant_points_by_DFS(
                    split_vectors[ind], split_edges[ind])
                inter[ind] = inter_area_redundant
        return inter

    def sphIoU(self, dets, gt, is_aligned=False, eps=1e-8):
        #! This program need high-precision float operator!!
        dets, gt = torch.deg2rad(dets).double(), torch.deg2rad(gt).double()

        '''Unbiased Spherical IoU Computation'''
        d_size, g_size = dets.shape[0], gt.shape[0]
        if is_aligned:
            res = torch.cat([dets, gt], dim=1)
        else:
            res = torch.hstack((dets.repeat_interleave(g_size, dim=0), gt.repeat(
            (d_size, 1)))).reshape(d_size * g_size, -1)
        
        area_A = self.area(res[:, 2], res[:, 3])
        area_B = self.area(res[:, 6], res[:, 7])

        inter = self.computeInter(res[:, :4], res[:, 4:])
        final = inter / (area_A + area_B - inter + eps)
        final = final if is_aligned else final.reshape(d_size, g_size)
        return final.float()

@torch.no_grad()
def transFormat(gt):
    '''
    Change the format and range of the Spherical Rectangle Representations.
    Itorchut:
    - gt: the last dimension: [center_x, center_y, fov_x, fov_y]
          center_x : [-180, 180]
          center_y : [90, -90]
          fov_x    : [0, 180]
          fov_y    : [0, 180]
          All parameters are angles.
    Output:
    - ann: the last dimension: [center_x', center_y', fov_x', fov_y']
           center_x' : [0, 2 * pi]
           center_y' : [0, pi]
           fov_x'    : [0, pi]
           fov_y'    : [0, pi]
           All parameters are radians.
    '''
    import copy
    ann = gt#copy.deepcopy(gt)
    ann[..., 2] = ann[..., 2] / 180 * torch.pi
    ann[..., 3] = ann[..., 3] / 180 * torch.pi
    ann[..., 0] = ann[..., 0] / 180 *torch.pi+ torch.pi
    ann[..., 1] = torch.pi / 2 - ann[..., 1] / 180 * torch.pi
    return ann
    

if __name__ == '__main__':
    '''
    Some Unbiased Spherical IoU Computation Examples.
    Note: the itorchut range for pred and gt (angles)
          [center_x, center_y, fov_x, fov_y]
          center_x : [-180, 180]
          center_y : [90, -90]
          fov_x    : [0, 180]
          fov_y    : [0, 180]
    The itorchut range for our unbiased IoU: (radians)
          [center_x, center_y, fov_x, fov_y]
          center_x : [0, 360]
          center_y : [0, 180]
          fov_x    : [0, 180]
          fov_y    : [0, 180]
    We use "transFormat" function to change the format.
    '''

    pred = torch.tensor([
        [-33.76, 43.99, 61.65, 43.88],
        [-51.96, 19.61, 61.65, 43.88],
        [-88.25, 7.33, 61.65, 43.88],
        [-109.89, -13.51, 61.65, 43.88],
        [-60.44, 14.7, 61.65, 43.88],
        [0, 0, 61.65, 43.88]
    ], requires_grad=True)

    gt = torch.tensor([
        [-37.9, 19.33, 64.09, 48.89],
        [-75.68, 12.1, 64.09, 48.89],
        [-97.17, -8.95, 64.09, 48.89],
        [-51.24, -29.18, 40.65, 42.58],
        [0, -1, 58.42, 40.32]
    ], requires_grad=True)

    _gt = transFormat(gt)
    _pred = transFormat(pred)
    
    eps=1e-6
    _sphIoU = Sph().sphIoU(_pred, _gt)
    
    loss = 1 - _sphIoU.mean()
    loss.backward()
    
    # Present the values for the IoU  calculation results in float format.
    torch.set_printoptions(precision=8, sci_mode=False)
    print(_sphIoU, _sphIoU.grad)
    