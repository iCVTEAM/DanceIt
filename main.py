import torch
from torch import optim
from model import *
import os
import json
from data import TrainSet
from Visualize_norm import visualizeKeypoints
from torch.utils.data import DataLoader
import argparse
import logging
import collections
import scipy.io.wavfile as wav
from python_speech_features import mfcc
import numpy as np

logging.basicConfig()
log = logging.getLogger("Running:")
log.setLevel(logging.DEBUG)
torch.manual_seed(1234)
np.random.seed(1234)


class AudoToDance(object):
    def __init__(self, args, is_test=False):
        super(AudoToDance, self).__init__
        self.is_test_mode = is_test
        self.train_data = TrainSet(args.data, args, train=True)
        self.val_data = TrainSet(args.data, args, train=False)
        self.train_dataloader = DataLoader(self.train_data, batch_size=args.batch_size, shuffle=True, drop_last=True)
        self.val_dataloader = DataLoader(self.val_data, batch_size=args.batch_size, drop_last=True)
        self.model = Match_Net(args).cuda(args.device)
        self.optim = optim.Adam(self.model.parameters(), lr=args.lr)
        self.loc_data = TrainSet('truth_data.json', args, test=True)
        self.loc_dataloader = DataLoader(self.loc_data, batch_size=1, shuffle=False)
        self.args = args
        if self.is_test_mode:
            self.test_data = TrainSet(args.data, args, test=True)
            self.test_dataloader = DataLoader(self.test_data, batch_size=1, shuffle=False)
            self.loadModelCheckpoint(args.test_model)

    def buildLoss(self, output, target):
        loss = (output-target)**2
        return torch.mean(loss)

    def saveModel(selfself, state_info, path):
        torch.save(state_info, path)

    def loadModelCheckpoint(self, path):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optim.load_state_dict(checkpoint['optim_state_dict'])

    def trainModel(self, max_epochs, logfldr):
        log.debug("Training Model")
        epoch_n = max_epochs
        best_train_loss, best_val_loss = float('inf'), float('inf')
        best_train_acc, best_val_acc = 0, 0
        iters_without_improvement = 0
        Loss_list = []
        for epoch in range(epoch_n):
            running_loss = 0.0
            accuracy = 0.0
            print("Epoch {}/{}".format(epoch, epoch_n))
            print("-" * 10)
            i = 0
            for data in self.train_dataloader:
                self.optim.zero_grad()

                train_x, train_y = data
                predictions, _, _ = self.model(train_x.cuda().float())
                loss = self.buildLoss(predictions.float(), train_y.cuda().float())
                loss = loss.cuda()
                loss.backward()
                self.optim.step()

                running_loss += loss.data.item()
                acc = len(predictions[predictions < 1])/self.args.batch_size
                accuracy += acc
            print("Loss is:{:.4f}".format(running_loss / len(self.train_dataloader)))
            print("accuracy is {:.4f}".format(accuracy / len(self.train_dataloader)))
            path = os.path.join(logfldr, "Epoch_{}".format(epoch))
            os.makedirs(path)
            path = os.path.join(path, "model_db.pth")
            state_info = {
                'epoch': epoch,
                'epoch_losses': loss,
                'model_state_dict': self.model.state_dict(),
                'optim_state_dict': self.optim.state_dict(),
            }
            self.saveModel(state_info, path)
            Loss_list.append(running_loss / len(self.train_dataloader))
            if best_train_loss > (running_loss / len(self.train_dataloader)):
                best_train_loss = (running_loss / len(self.train_dataloader))
                path = os.path.join(logfldr, "best_model_db.pth")
                self.saveModel(state_info, path)
                iters_without_improvement = 0
            else:
                iters_without_improvement += 1
                if iters_without_improvement >= 100:
                    log.info("Stopping Early because no improvment in {}".format(
                        iters_without_improvement))
                    break
            if best_train_acc < (accuracy / len(self.train_dataloader)):
                best_train_acc = (accuracy / len(self.train_dataloader))
            if epoch % 10 == 0:
                val_loss = 0.0
                val_acc = 0.0
                print("Epoch {}/{}".format(epoch, epoch_n))
                print("-" * 10)

                for data in self.val_dataloader:
                    val_x, val_y = data
                    predictions, _, _ = self.model(train_x.cuda().float())
                    vloss = self.buildLoss(predictions, train_y.cuda().float())
                    vloss = vloss.cuda()

                    val_loss += vloss.data.item()
                    acc = len(predictions[predictions < 1])/self.args.batch_size
                    val_acc += acc 
                print("val Loss is:{:.4f}".format(val_loss / len(self.val_dataloader)))
                print("val_accuracy is {:.4f}".format(val_acc / len(self.val_dataloader)))
                if best_val_loss > (val_loss / len(self.val_dataloader)):
                    best_val_loss = (val_loss / len(self.val_dataloader))
                if best_val_acc < (val_acc / len(self.val_dataloader)):
                    best_val_acc = (val_acc / len(self.val_dataloader))
        np.save('loss_1.npy',Loss_list)
        #x1 = range(500)
        #plt.plot(x1,Loss_list)
        #plt.xlabel('Train loss vs epoches')
        #plt.savefig("accuracy_loss.jpg")
        return best_train_loss, best_val_loss, best_train_acc, best_val_acc
        # for parameters in self.model.parameters():
        # print(parameters)

    def LocKeyJSON(self, logfldr):

        log.debug("Loc Keyps_Feature")
        keyps_feature = []
        keyps_data = {}
        i = 0

        for data in self.loc_dataloader:
            loc_x, loc_y = data
            _, _, predictions = self.model(loc_x.cuda().float())
            predictions = np.array(predictions.cpu().detach().numpy())
            predictions = predictions.flatten()
            predictions = predictions.tolist()
            keyps_data[i] = predictions
            i += 1

        json.dumps(keyps_data)
        with open(logfldr + '/keyps_feature.json', 'a') as t:
            json.dump(keyps_data, t)

    def testModel(self, args, logfldr):
        log.debug("Test Model:")
        with open(logfldr + '/keyps_feature.json', 'r+') as t:
            data_keyps = json.load(t)

        with open("truth_data.json", "r+") as f:
            truth_keyps = json.load(f, object_pairs_hook=collections.OrderedDict)

        ans = []
        loss = 0
        tem = -1
        if args.audio_file is None:
            for data in self.test_dataloader:
                test_x, _ = data
                _, predictions, _ = self.model(test_x.cuda().float())
                min = float('inf')
                loc_index = 0
                for index, keyps_feature in data_keyps.items(): 
                    if index == tem: continue
                    keyps_feature = np.array(keyps_feature)
                    keyps_feature = torch.from_numpy(keyps_feature).cuda()
                    distance = self.buildLoss(keyps_feature.double(), predictions.double())
                    if min > distance:
                        loc_index = index
                        min = distance
                target = truth_keyps['0'][0][int(loc_index)][args.len_seg * 24:]
                tem = loc_index
                target = np.array(target)
                ans.append(target)
                loss = max
            log.debug("Write video:")
            vid_path = "{}\{}.mp4".format(logfldr, 'result')
            visualizeKeypoints(ans, vid_path)
            return loss, loss
        else:
            (rate, sig) = wav.read(args.audio_file)
            mfcc_feat = mfcc(sig, rate, winlen=0.20, winstep=0.08, numcep=24)
            mfcc_feat = mfcc_feat.reshape(-1,)
            len_music = len(mfcc_feat)//(24*args.len_seg)
            mfcc_feat = mfcc_feat[:len_music*24*args.len_seg]
            audio_mfcc = np.split(mfcc_feat,len_music)
            for audio_keyps in audio_mfcc:
                audio_keyps = np.reshape(audio_keyps,(1,args.len_seg,24))
                audio_keyps = torch.from_numpy(audio_keyps)
                predictions = self.model(audio_keyps.float().cuda())
                min = float('inf')
                loc_index = 0
                for index, keyps_feature in data_keyps.items(): 
                    if index == tem: continue
                    keyps_feature = np.array(keyps_feature)
                    keyps_feature = torch.from_numpy(keyps_feature).cuda()
                    distance = self.buildLoss(keyps_feature.double(), predictions.double())
                    if min > distance:
                        loc_index = index
                        min = distance
                target = truth_keyps[str(int(loc_index))][0][args.len_seg * 24:]
                tem = loc_index
                target = np.array(target)
                ans.append(target)
                loss = max
            log.debug("Write video:")
            vid_path = "{}/{}.mp4".format('video', 'result')
            visualizeKeypoints(ans, vid_path, args.audio_file, args)
            return loss, loss    


def createOptions():
    parser = argparse.ArgumentParser(
        description="Pytorch: Audio To Everybody Dance Model"
    )
    parser.add_argument("--data", type=str, default='train_data.json')
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--test_model", type=str, default=None)
    parser.add_argument("--logfldr", type=str, default='checkpoint')
    parser.add_argument("--batch_size", type=int, default=50)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--max_epoch", type=int, default=500)
    parser.add_argument("--train_audio", type=bool, default=True)
    parser.add_argument("--audio_file", type=str, default=None)
    parser.add_argument("--len_seg", type=int, default=100)
    parser.add_argument("--num_node", type=int, default=23)

    args = parser.parse_args()
    return args


def main():
    args = createOptions()
    args.device = torch.device(args.device)
    is_test_mode = args.test_model is not None
    if is_test_mode:
        args.batch_size = 1

    dance_learner = AudoToDance(args, is_test=is_test_mode)

    logfldr = args.logfldr
    if not os.path.isdir(logfldr):
        os.makedirs(logfldr)

    if not is_test_mode:
        min_train, min_val, min_train_acc, min_val_acc = dance_learner.trainModel(args.max_epoch, logfldr)
        args.batch_size = 1
        dance_learner = AudoToDance(args, is_test=is_test_mode)
        dance_learner.loadModelCheckpoint('checkpoint/best_model_db.pth')
        dance_learner.LocKeyJSON(logfldr)
    else:
        min_train, min_val = dance_learner.testModel(args, logfldr)
    best_lossess = [min_train, min_val]
    log.info("The best validation is {}".format(best_lossess))


if __name__ == '__main__':
    main()
