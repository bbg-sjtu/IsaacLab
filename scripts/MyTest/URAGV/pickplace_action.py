import torch
# from IKCtrl import URAGV_IK
import random

pick_x=-0.9
def getPickplaceAction(rigid_obj_pos,container_pos):
    obj_x=rigid_obj_pos[0][0].item()
    obj_y=rigid_obj_pos[0][1].item()
    obj_z=rigid_obj_pos[0][2].item()
    container_x=container_pos[0][0].item()

    # 运动至第一个container（根据obj确定y坐标），并打开手爪
    handaction=hand_action(True)
    agv_y=obj_y+random.uniform(-0.1, 0.1) # y±0.1
    agv_arm_action=torch.tensor([
        # agv
        pick_x,  # x
        agv_y, # y
        0.0, # z

        # arm default
        3.14,
        -1.57,
        1.57,
        -3.14,
        -1.57,
        3.14,
    ]).unsqueeze(0)
    action1 = torch.cat((agv_arm_action,handaction), dim=1)

    # 控制机械臂末端运动到物体上方，打开手爪
    ee_goals = torch.tensor([[obj_x, obj_y, obj_z+0.1, 0, -0.7071,  0, 0.7071]])
    base_action=torch.tensor([[pick_x,agv_y,0.0]])
    handaction=hand_action(True)
    action2 = torch.cat((base_action,ee_goals,handaction), dim=1)

    # 控制机械臂末端运动到物体处，打开手爪
    ee_goals = torch.tensor([[obj_x, obj_y, obj_z+0.05, 0, -0.7071,  0, 0.7071]])
    base_action=torch.tensor([[pick_x,agv_y,0.0]])
    handaction=hand_action(True)
    action3 = torch.cat((base_action,ee_goals,handaction), dim=1)

    # 控制机械臂末端运动到物体处，关闭手爪
    ee_goals = torch.tensor([[obj_x, obj_y, obj_z+0.05, 0, -0.7071,  0, 0.7071]])
    base_action=torch.tensor([[pick_x,agv_y,0.0]])
    handaction=hand_action(False)
    action4 = torch.cat((base_action,ee_goals,handaction), dim=1)

    # 将物体提起
    ee_goals = torch.tensor([[obj_x, obj_y, obj_z+0.2, 0, -0.7071,  0, 0.7071]])
    base_action=torch.tensor([[pick_x,agv_y,0.0]])
    handaction=hand_action(False)
    action5 = torch.cat((base_action,ee_goals,handaction), dim=1)

    handaction=hand_action(False)
    agv_arm_action=torch.tensor([
        # agv
        pick_x, # x
        agv_y, # y
        0.0, # z

        # arm default
        3.14,
        -1.57,
        1.57,
        -3.14,
        -1.57,
        3.14,
    ]).unsqueeze(0)
    action6 = torch.cat((agv_arm_action,handaction), dim=1)
    

    # 运动到第二个container
    handaction=hand_action(False)
    agv_x=container_x+random.uniform(-0.1, 0.1) # x±0.1
    agv_arm_action=torch.tensor([
        # agv
        agv_x, # x
        0.1, # y
        0.0, # z

        # arm default
        1.57,
        -1.57,
        1.57,
        -3.14,
        -1.57,
        3.14,
    ]).unsqueeze(0)
    action7 = torch.cat((agv_arm_action,handaction), dim=1)

    # 打开手爪
    handaction=hand_action(True)
    agv_arm_action=torch.tensor([
        # agv
        agv_x,  # x
        0.1, # y
        0.0, # z

        # arm default
        1.57,
        -1.57,
        1.57,
        -3.14,
        -1.57,
        3.14,
    ]).unsqueeze(0)
    action8 = torch.cat((agv_arm_action,handaction), dim=1)

    actions=[action1,action2,action3,action4,action5,action6,action7,action8]
    # actions_gripperbool=getPickplaceAction_withbool(rigid_obj_pos,container_pos)
    return actions

def hand_action(open:bool):
    if open==True: 
        # 打开
        handaction=torch.tensor([
            0.03,
            0.03
        ]).unsqueeze(0)
    else:
        # 关闭
        handaction=torch.tensor([
            0.0, 
            0.0,
        ]).unsqueeze(0)
    return handaction