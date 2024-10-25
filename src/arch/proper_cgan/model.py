from arch.cgan.utils import device


class Generator:
    def __init__(
        self,
        device=device(),
    ):

        self.net = self.build_G_net()
        if G_net is None:
            self.G_net = self.build_G_net(n_input=1, n_output=2, size=256, body=body)
        else:
            self.G_net = G_net

        if optimizer == "Adam":
            self.optimizer = optim.Adam(self.G_net.parameters(), lr=0.0004)

        self.criterion = nn.L1Loss()

    def build_G_net(self, n_input=1, n_output=2, size=256, body="resnet34"):
        if body == "resnet34":
            body_model = resnet34()
        backbone = create_body(body_model, pretrained=True, n_in=n_input, cut=-2)
        G_net = DynamicUnet(backbone, n_output, (size, size)).to(self.device)
        return G_net

    def pretrain(self, train_dl, epochs):
        for itr in range(epochs):
            loss_meter = AverageMeter()
            for data in tqdm(train_dl):
                L, ab = data["L"].to(self.device), data["ab"].to(self.device)
                preds = self.G_net(L)
                loss = self.criterion(preds, ab)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                loss_meter.update(loss.item(), L.size(0))

            print(f"Epoch {itr + 1}/{epochs}")
            print(f"L1 Loss: {loss_meter.avg:.5f}")

    def get_model(self):
        return self.G_net

    def save_model(self, path="generator.pt"):
        torch.save(self.G_net.state_dict(), path)

    def load_model(self, path="generator.pt"):
        self.G_net.load_state_dict(torch.load(path), map_location=self.device)
