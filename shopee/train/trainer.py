

class Trainer:

    def __init__(
        self,
        model,
        train_dataloader,
        val_dataloader,
        loss_fn,
        optimizer,
        scheduler,
        device="cpu",
        writer,
    ):
    self.model = model
    self.train_dataloader = train_dataloader
    self.val_dataloader = val_dataloader
    self.loss_fn = loss_fn
    self.optimizer = optimizer
    self.scheduler = scheduler
    self.device = device
    self.writer = writer


def train_epoch(train_dataloader, model, loss_fn, epoch, optimizer, scheduler, device="cpu"):

    # train
    self.model.train()
    self.model.to(device)
    losses = []
    total_loss = []
    total_lr = []
    total_samples = 0

    for batch_idx, (triplet, label) in enumerate(train_dataloader):
#         image = [img.to(device) for img in image]
#         text = [txt.to(device) for txt in text]
#         triplet = tuple(zip(image, text))
        triplet = [(img.to(device), txt.to(device)) for (img, txt) in triplet]
#         print(triplet)
        label = tuple(l.to(device) for l in label)

        optimizer.zero_grad()
        outputs = self.model(*triplet)

        loss_inputs = outputs

        loss_outputs = loss_fn(loss_inputs, label)
        loss = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs
        losses.append(loss.item())
        total_loss.append(loss.item())
        total_lr.append(scheduler.get_last_lr())
        loss.backward()
        optimizer.step()
        scheduler.step()

        count = batch_idx + 1
        total_samples += len(triplet[0][0])

        if count % log_interval == 0 or count == len(train_dataloader):
            mean_loss = np.mean(losses)
            message = 'Train: [{: 6d}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                total_samples, len(train_dataloader.dataset),
                100. * count / len(train_dataloader), mean_loss)

            print(message)
            losses = []

            writer.add_scalar(
                "train/loss", 
                mean_loss, 
                global_step=count / len(train_dataloader) + epoch * len(train_dataloader)
            )

    return total_lr, total_loss

    def val_epoch(
    val_dataloader, 
    model, 
    labels, 
    epoch, 
    device="cpu"
):
    with torch.no_grad():
        self.model.eval()
        embeddings = []
        total_samples = 0

        for batch_idx, batch in enumerate(val_dataloader):
            batch = [b.to(device) for b in batch]
            embeddings.append(self.model(batch).squeeze().cpu())
            count = batch_idx + 1
            total_samples += len(batch[0])

        embeddings = torch.cat(embeddings).numpy()

        thresholds = np.linspace(0.0, 1.0, num=101)
        scores = compute_f1_score(embeddings, labels, thresholds=thresholds)
        best = np.argmax(scores)
        best_threshold = thresholds[best]
        best_score = scores[best]

        del embeddings # for avoiding memory issues
        gc.collect()


        message = 'Val: [{: 6d}/{} ({:.0f}%)]\tBest F1: {:.6f} (threshold: {:.2f})'.format(
            total_samples, len(val_dataloader.dataset),
            100. * count / len(val_dataloader), 
            best_score, best_threshold,
        )

        print(message)

        writer.add_scalar("val/f1", best_score, global_step=epoch + 1)
        writer.add_scalar("val/threshold", best_threshold, global_step=epoch + 1)

        return best_score