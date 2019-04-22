import cv2
import numpy as np
import copy
from scipy import stats

def lines_extract_from_depth(img):
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #print(gray.shape)
    else:
        gray = img
    kernel = np.ones((15,15), np.uint8)
    mclose = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)

    edges = cv2.Canny(mclose, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, 1, 10)

    return lines

def plot_histogram(x):
    import matplotlib.pyplot as plt
    n, bins, patches = plt.hist(x=x, bins=36, range=(-np.pi, np.pi),  color='#0504aa', alpha=0.7, rwidth=0.85)
    plt.grid(axis='y', alpha=0.75)
    plt.xlabel('The angle between the lines and x axis.')
    plt.ylabel('Frequency')
    plt.text(23, 45, r'$\mu=15, b=3$')
    maxfreq = n.max()
    plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)
    plt.show()

def lines_detector(img):
    mask, _ = get_roi_hsv(img)
    edges = cv2.Canny(mask, 50, 150)
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=30, minLineLength=40, maxLineGap=10)
    if lines is not None:
        lines = np.squeeze(lines, axis = 1)
    return lines


def lines_filter(lines):
    cnts, edgs = np.histogram(lines[:, 1], bins=36, range=(-np.pi, np.pi))
    #plot_histogram(lines[:, 1])
    #print(cnts)
    #print(edgs)

    #find the most occured angle
    cnt_max_ind = np.argmax(cnts)
    if cnt_max_ind + 1 > len(cnts):
        raise IndexError, "index out of array length"
    cad_ind = (lines[:, 1] >= edgs[cnt_max_ind]) * (lines[:, 1] <= edgs[cnt_max_ind + 1]) 
    edg_max = edgs[cnt_max_ind]
    res_lines = lines[cad_ind, :]
    
    # find if origin point exist between lines
    for i in np.where(cnts>0)[0]:
        if abs(abs(edgs[i] - edg_max) - np.pi) <= np.pi / 12:
            if i + 1 > len(cnts):
                raise IndexError, "index out of array length"
            cad_ind2 = (lines[:, 1] >= edgs[i]) * (lines[:, 1] <= edgs[i + 1]) 
            res_lines2 = lines[cad_ind2, :]
            return res_lines, res_lines2
    return res_lines
        

def calc_cenline(lines):
    """lines is an array of polar coordinates"""
    cenline_rho = None
    cenline_theta = None
    if isinstance(lines, tuple):
        l1, l2 = lines
        l1_max_index = np.argmax(l1[:, 0])
        l2_max_index = np.argmax(l2[:, 0])
        l1_max_rho, l1_max_theta = l1[l1_max_index]
        l2_max_rho, l2_max_theta = l2[l2_max_index]
        if l1_max_rho + l2_max_rho > 30 and l1_max_rho + l2_max_rho < 60:
            cenline_rho = abs(l2_max_rho - l1_max_rho) / 2
            #cenline_theta = (l1_max_theta + np.pi + l2_max_theta) / 2
            cenline_theta = l1_max_theta if l1_max_rho > l2_max_rho else l2_max_theta
    else:             
        #adjust results after filtering 
        far_index = np.argmax(lines[:, 0])
        near_index = np.argmin(lines[:, 0])
        if (lines[far_index, 0] - lines[near_index, 0]) > 30 and (lines[far_index, 0] - lines[near_index, 0]) < 60:
            cenline_rho = (lines[far_index, 0] + lines[near_index, 0]) / 2
            #cenline_theta = np.mean(lines[:, 1])
            cenline_theta = (lines[far_index, 1] + lines[near_index, 1]) / 2
            #cenline_theta = np.mean(lines[:, 1])
    return cenline_rho, cenline_theta

    

def cenline_extract(img, ret_type = "polar"):
    """
    This function use hough transform and histogram to extract the center line of the pipe.
    ret_type: "polar"--return (rho, theta) in polar coordinate; "points": return (x1, y1, x2, y2)the two points on the line
    """
    #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #gray = cv2.GaussianBlur(img, (3,3), 0)
    lines = lines_detector(img)

    if lines is not None: 
        lines_polar = [list(line_xy2polar(l[:2], l[2:])) for l in lines]
        lines_polar = np.array(lines_polar)   
        
        # filter some noise lines
        #print(lines_polar[:, 1])
       
        lines_polar = lines_filter(lines_polar)
        cenline_rho, cenline_theta = calc_cenline(lines_polar)
        if cenline_rho is None or cenline_theta is None:
            return None
        if ret_type == "polar":
            return cenline_rho, cenline_theta
        else:
            x1,y1,x2,y2 = line_polar2xy(cenline_rho, cenline_theta)
            return x1, y1, x2, y2
    else:
        return None

def line_xy2polar(p1, p2):
    '''turn the line defined by two points p1,p2 into polar coordinate'''
    x1, y1 = p1
    x2, y2 = p2
    rp = np.square(y1 - y2) + np.square(x2 - x1)
    tmp = x1 * y2 - x2 * y1
    x = tmp * (y2 - y1) / rp
    y = -tmp * (x2 - x1) / rp
    # (x,y) is the normal vector of the line
    rho = np.sqrt(x**2 + y**2)
    if x == 0 and y == 0:
        theta = np.pi / 2 + np.arctan2(y2-y1, x1-x2)
        theta = np.arctan2(np.sin(theta), np.cos(theta))
    else:
        theta = np.arctan2(y, x)
    return (rho, theta)

def dist_p2line(p, p1, p2):
    """calculate the distance from point p to line defined by p1 and p2"""
    x0, y0 = p
    x1, y1 = p1
    x2, y2 = p2
    rp = np.sqrt(np.square(x1-x2) + np.square(y1-y2))
    dist = abs((y2-y1) * x0 - (x2 - x1)*y0 + x2*y1-x1*y2) / rp
    return dist

def line_polar2xy(rho, theta):
    #print(rho)
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a*rho
    y0 = b*rho
    x1 = int(x0 + 1000*(-b))
    y1 = int(y0 + 1000*(a))
    x2 = int(x0 - 1000*(-b))
    y2 = int(y0 - 1000*(a)) 
    return (x1, y1, x2, y2)

def get_reward(img, u):
    """
    The function for calculate reward. u is the forward velocity of the auv
    The definition of reward:
            
                    r = u * (|cos(theta)| + 1 - d/max_d)    
    """
    try:
        extract_res = cenline_extract(img, "polar")
        if extract_res is not None:
            rho, theta = extract_res
            x1, y1, x2, y2 = line_polar2xy(rho, theta)
            height, width = img.shape[:2]
            d = dist_p2line((width/2, height/2), (x1, y1), (x2, y2))
            max_d = np.sqrt((width/2)**2 + (height/2)**2)
            #rew = -u * (abs(np.cos(theta)) + 1 - d*1.0 / max_d)
            trav_dist = -u * abs(np.cos(theta))           #traveled distance along the line
            perp_dist_penalty = -d * 3.0 / width if d < width / 3.0 else -1
            
            valid_dir_bonus = 1 if abs(np.cos(theta)) > np.cos(np.pi/9) else 0
            #print "reward terms:", trav_dist, ",", perp_dist_penalty, ",", valid_dir_bonus
            #rew = 10 * trav_dist + 1.0 * perp_dist_penalty + valid_dir_bonus
            #rew = 1 * trav_dist + 10 * perp_dist_penalty 
            #rew = 1.0 * trav_dist/(d * 3 / width + 0.1) + 10 * perp_dist_penalty
            rew = 1.0 * trav_dist
            return rew, [np.cos(theta), d, rew/u, u]
        else:
            return None, None
    except IndexError:
        return None, None

def get_roi_hsv(img, lower=np.array([60, 43, 46]), upper = np.array([70, 255, 255])):
    # change to hsv space
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # get mask for roi 
    mask = cv2.inRange(hsv, lower, upper)
    # close operation 
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    #mask = cv2.dilate(mask, kernel)
    res = cv2.bitwise_and(img, img, mask=mask)
    return mask, res



if __name__ == "__main__":
    img = cv2.imread("/home/uwsim/workspace/results/pipeline_track/record2/img10.jpg")
    print(img.shape)
    res = cenline_extract(img, "points")
    print(get_reward(img, 1))

    if res is not None:
        x1, y1, x2, y2 = res
        cv2.line(img, (x1,y1), (x2, y2), (0, 0, 255), 2)
        #print(img.shape)
        cv2.imshow("o", img)
        cv2.waitKey(0)
        cv2.imwrite("tmp_results/cen_line.jpg", img)
    
    
